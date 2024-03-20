import torch
import clip
import models.vqvae as vqvae
from models.vqvae_sep import VQVAE_SEP
import models.t2m_trans as trans
import models.t2m_trans_uplow as trans_uplow
import numpy as np
from exit.utils import visualize_2motions
import options.option_transformer as option_trans



##### ---- CLIP ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# https://github.com/openai/CLIP/issues/111
class TextCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        with torch.no_grad():
            word_emb = self.model.token_embedding(text).type(self.model.dtype)
            word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.model.transformer(word_emb)
            word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
            enctxt = self.model.encode_text(text).float()
        return enctxt, word_emb
clip_model = TextCLIP(clip_model)

def get_vqvae(args, is_upper_edit):
    if not is_upper_edit:
        return vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                            args.nb_code,
                            args.code_dim,
                            args.output_emb_width,
                            args.down_t,
                            args.stride_t,
                            args.width,
                            args.depth,
                            args.dilation_growth_rate)
    else:
        return VQVAE_SEP(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate,
                        moment={'mean': torch.from_numpy(args.mean).cuda().float(), 
                            'std': torch.from_numpy(args.std).cuda().float()},
                        sep_decoder=True)

def get_maskdecoder(args, vqvae, is_upper_edit):
    tranformer = trans if not is_upper_edit else trans_uplow
    return tranformer.Text2Motion_Transformer(vqvae,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)

class MMM(torch.nn.Module):
    def __init__(self, args=None, is_upper_edit=False):
        super().__init__()
        self.is_upper_edit = is_upper_edit


        args.dataname = args.dataset_name = 't2m'

        self.vqvae = get_vqvae(args, is_upper_edit)
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        self.vqvae.load_state_dict(ckpt['net'], strict=True)
        if is_upper_edit:
            class VQVAE_WRAPPER(torch.nn.Module):
                def __init__(self, vqvae) :
                    super().__init__()
                    self.vqvae = vqvae
                    
                def forward(self, *args, **kwargs):
                    return self.vqvae(*args, **kwargs)
            self.vqvae = VQVAE_WRAPPER(self.vqvae)
        self.vqvae.eval()
        self.vqvae.cuda()

        self.maskdecoder = get_maskdecoder(args, self.vqvae, is_upper_edit)
        ckpt = torch.load(args.resume_trans, map_location='cpu')
        self.maskdecoder.load_state_dict(ckpt['trans'], strict=True)
        self.maskdecoder.train()
        self.maskdecoder.cuda()

    def forward(self, text, lengths=-1, rand_pos=True):
        b = len(text)
        feat_clip_text = clip.tokenize(text, truncate=True).cuda()
        feat_clip_text, word_emb = clip_model(feat_clip_text)
        index_motion = self.maskdecoder(feat_clip_text, word_emb, type="sample", m_length=lengths, rand_pos=rand_pos, if_test=False)

        m_token_length = torch.ceil((lengths)/4).int()
        pred_pose_all = torch.zeros((b, 196, 263)).cuda()
        for k in range(b):
            pred_pose = self.vqvae(index_motion[k:k+1, :m_token_length[k]], type='decode')
            pred_pose_all[k:k+1, :int(lengths[k].item())] = pred_pose
        return pred_pose_all
    

if __name__ == '__main__':
    args = option_trans.get_args_parser()

# python generate.py --resume-pth '/home/epinyoan/git/MaskText2Motion/T2M-BD/output/vq/2023-07-19-04-17-17_12_VQVAE_20batchResetNRandom_8192_32/net_last.pth' --resume-trans '/home/epinyoan/git/MaskText2Motion/T2M-BD/output/t2m/2023-10-12-10-11-15_HML3D_45_crsAtt1lyr_40breset_WRONG_THIS_20BRESET/net_last.pth' --text 'the person crouches and walks forward.' --length 156

    mmm = MMM(args).cuda()
    pred_pose = mmm([args.text], torch.tensor([args.length]).cuda(), rand_pos=False)

    std = np.load('./exit/t2m-std.npy')
    mean = np.load('./exit/t2m-mean.npy')
    file_name = '_'.join(args.text.split(' '))+'_'+str(args.length)
    visualize_2motions(pred_pose[0].detach().cpu().numpy(), std, mean, 't2m', args.length, save_path='./output/'+file_name+'.html')


