import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from models.t2m_trans import Decoder_Transformer, Encoder_Transformer
from exit.utils import generate_src_mask
import torch
from utils.humanml_utils import HML_UPPER_BODY_MASK, HML_LOWER_BODY_MASK, UPPER_JOINT_Y_MASK

class VQVAE_SEP(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 moment=None,
                 sep_decoder=False):
        super().__init__()
        if args.dataname == 'kit':
            self.nb_joints = 21
            output_dim = 251
            upper_dim = 120        
            lower_dim = 131  
        else:
            self.nb_joints = 22
            output_dim = 263
            upper_dim = 156        
            lower_dim = 107 
        self.code_dim = code_dim
        if moment is not None:
            self.moment = moment
            self.register_buffer('mean_upper', torch.tensor([0.1216, 0.2488, 0.2967, 0.5027, 0.4053, 0.4100, 0.5703, 0.4030, 0.4078, 0.1994, 0.1992, 0.0661, 0.0639], dtype=torch.float32))
            self.register_buffer('std_upper', torch.tensor([0.0164, 0.0412, 0.0523, 0.0864, 0.0695, 0.0703, 0.1108, 0.0853, 0.0847, 0.1289, 0.1291, 0.2463, 0.2484], dtype=torch.float32))
        # self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        
        # self.encoder = Encoder(output_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.sep_decoder = sep_decoder
        if self.sep_decoder:
            self.decoder_upper = Decoder(upper_dim, int(code_dim/2), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        
            self.decoder_lower = Decoder(lower_dim, int(code_dim/2), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        
        else:
            self.decoder = Decoder(output_dim, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        


        self.num_code = nb_code

        self.encoder_upper = Encoder(upper_dim, int(code_dim/2), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_lower = Encoder(lower_dim, int(code_dim/2), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.quantizer_upper = QuantizeEMAReset(nb_code, int(code_dim/2), args)
        self.quantizer_lower = QuantizeEMAReset(nb_code, int(code_dim/2), args)

    def rand_emb_idx(self, x_quantized, quantizer, idx_noise):
        # x_quantized = x_quantized.detach()
        x_quantized = x_quantized.permute(0,2,1)
        mask = torch.bernoulli(idx_noise * torch.ones((*x_quantized.shape[:2], 1),
                                                device=x_quantized.device))
        r_indices = torch.randint(int(self.num_code/2), x_quantized.shape[:2], device=x_quantized.device)
        r_emb = quantizer.dequantize(r_indices)
        x_quantized = mask * r_emb + (1-mask) * x_quantized
        x_quantized = x_quantized.permute(0,2,1)
        return x_quantized
    
    def normalize(self, data):
        return (data - self.moment['mean']) / self.moment['std']
    
    def denormalize(self, data):
        return data * self.moment['std'] + self.moment['mean']
    
    def normalize_upper(self, data):
        return (data - self.mean_upper) / self.std_upper
    
    def denormalize_upper(self, data):
        return data * self.std_upper + self.mean_upper
    
    def shift_upper_down(self, data):
        data = data.clone()
        data = self.denormalize(data)
        shift_y = data[..., 3:4].clone()
        data[..., UPPER_JOINT_Y_MASK] -= shift_y
        _data = data.clone()
        data = self.normalize(data)
        data[..., UPPER_JOINT_Y_MASK] = self.normalize_upper(_data[..., UPPER_JOINT_Y_MASK])
        return data
    
    def shift_upper_up(self, data):
        _data = data.clone()
        data = self.denormalize(data)
        data[..., UPPER_JOINT_Y_MASK] = self.denormalize_upper(_data[..., UPPER_JOINT_Y_MASK])
        shift_y = data[..., 3:4].clone()
        data[..., UPPER_JOINT_Y_MASK] += shift_y
        data = self.normalize(data)
        return data
    
    def forward(self, x, *args, type='full', **kwargs):
        '''type=[full, encode, decode]'''
        if type=='full':
            x = x.float()
            x = self.shift_upper_down(x)

            upper_emb = x[..., HML_UPPER_BODY_MASK]
            lower_emb = x[..., HML_LOWER_BODY_MASK]
            upper_emb = self.preprocess(upper_emb)
            upper_emb = self.encoder_upper(upper_emb)
            upper_emb, loss_upper, perplexity = self.quantizer_upper(upper_emb)

            lower_emb = self.preprocess(lower_emb)
            lower_emb = self.encoder_lower(lower_emb)
            lower_emb, loss_lower, perplexity = self.quantizer_lower(lower_emb)
            loss = loss_upper + loss_lower

            if 'idx_noise' in kwargs and kwargs['idx_noise'] > 0:
                upper_emb = self.rand_emb_idx(upper_emb, self.quantizer_upper, kwargs['idx_noise'])
                lower_emb = self.rand_emb_idx(lower_emb, self.quantizer_lower, kwargs['idx_noise'])


            # x_in = self.preprocess(x)
            # x_encoder = self.encoder(x_in)
        
            # ## quantization
            # x_quantized, loss, perplexity  = self.quantizer(x_encoder)

            ## decoder
            if self.sep_decoder:
                x_decoder_upper = self.decoder_upper(upper_emb)
                x_decoder_upper = self.postprocess(x_decoder_upper)
                x_decoder_lower = self.decoder_lower(lower_emb)
                x_decoder_lower = self.postprocess(x_decoder_lower)
                x_out = merge_upper_lower(x_decoder_upper, x_decoder_lower)
                x_out = self.shift_upper_up(x_out)

            else:
                x_quantized = torch.cat([upper_emb, lower_emb], dim=1)
                x_decoder = self.decoder(x_quantized)
                x_out = self.postprocess(x_decoder)
            
            return x_out, loss, perplexity
        elif type=='encode':
            N, T, _ = x.shape
            x = self.shift_upper_down(x)

            upper_emb = x[..., HML_UPPER_BODY_MASK]
            upper_emb = self.preprocess(upper_emb)
            upper_emb = self.encoder_upper(upper_emb)
            upper_emb = self.postprocess(upper_emb)
            upper_emb = upper_emb.reshape(-1, upper_emb.shape[-1])
            upper_code_idx = self.quantizer_upper.quantize(upper_emb)
            upper_code_idx = upper_code_idx.view(N, -1)

            lower_emb = x[..., HML_LOWER_BODY_MASK]
            lower_emb = self.preprocess(lower_emb)
            lower_emb = self.encoder_lower(lower_emb)
            lower_emb = self.postprocess(lower_emb)
            lower_emb = lower_emb.reshape(-1, lower_emb.shape[-1])
            lower_code_idx = self.quantizer_lower.quantize(lower_emb)
            lower_code_idx = lower_code_idx.view(N, -1)

            code_idx = torch.cat([upper_code_idx.unsqueeze(-1), lower_code_idx.unsqueeze(-1)], dim=-1)
            return code_idx

        elif type=='decode':
            if self.sep_decoder:
                x_d_upper = self.quantizer_upper.dequantize(x[..., 0])
                x_d_upper = x_d_upper.permute(0, 2, 1).contiguous()
                x_d_upper = self.decoder_upper(x_d_upper)
                x_d_upper = self.postprocess(x_d_upper)

                x_d_lower = self.quantizer_lower.dequantize(x[..., 1])
                x_d_lower = x_d_lower.permute(0, 2, 1).contiguous()
                x_d_lower = self.decoder_lower(x_d_lower)
                x_d_lower = self.postprocess(x_d_lower)
            
                x_out = merge_upper_lower(x_d_upper, x_d_lower)
                x_out = self.shift_upper_up(x_out)
                return x_out
            else:
                x_d_upper = self.quantizer_upper.dequantize(x[..., 0])
                x_d_lower = self.quantizer_lower.dequantize(x[..., 1])
                x_d = torch.cat([x_d_upper, x_d_lower], dim=-1)
                x_d = x_d.permute(0, 2, 1).contiguous()
                x_decoder = self.decoder(x_d)
                x_out = self.postprocess(x_decoder)
                return x_out

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x
    
    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


def merge_upper_lower(upper_emb, lower_emb):
    motion = torch.empty(*upper_emb.shape[:2], 263).to(upper_emb.device)
    motion[..., HML_UPPER_BODY_MASK] = upper_emb
    motion[..., HML_LOWER_BODY_MASK] = lower_emb
    return motion

def upper_lower_sep(motion, joints_num):
    # root
    _root = motion[..., :4] # root

    # position
    start_indx = 1 + 2 + 1
    end_indx = start_indx + (joints_num - 1) * 3
    positions = motion[..., start_indx:end_indx]
    positions = positions.view(*motion.shape[:2], (joints_num - 1), 3)

    # 6drot
    start_indx = end_indx
    end_indx = start_indx + (joints_num - 1) * 6
    _6d_rot = motion[..., start_indx:end_indx]
    _6d_rot = _6d_rot.view(*motion.shape[:2], (joints_num - 1), 6)

    # joint_velo
    start_indx = end_indx
    end_indx = start_indx + joints_num * 3
    joint_velo = motion[..., start_indx:end_indx]
    joint_velo = joint_velo.view(*motion.shape[:2], joints_num, 3)

    # foot_contact
    foot_contact = motion[..., end_indx:]

    ################################################################################################
    #### Lower Body
    if joints_num == 22:
        lower_body = torch.tensor([0,1,2,4,5,7,8,10,11])
    else:
        lower_body = torch.tensor([0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    lower_body_exclude_root = lower_body[1:] - 1

    LOW_positions = positions[:,:, lower_body_exclude_root].view(*motion.shape[:2], -1)
    LOW_6d_rot = _6d_rot[:,:, lower_body_exclude_root].view(*motion.shape[:2], -1)
    LOW_joint_velo = joint_velo[:,:, lower_body].view(*motion.shape[:2], -1)
    lower_emb = torch.cat([_root, LOW_positions, LOW_6d_rot, LOW_joint_velo, foot_contact], dim=-1)

    #### Upper Body
    if joints_num == 22:
        upper_body = torch.tensor([3,6,9,12,13,14,15,16,17,18,19,20,21])
    else:
        upper_body = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    upper_body_exclude_root = upper_body - 1

    UP_positions = positions[:,:, upper_body_exclude_root].view(*motion.shape[:2], -1)
    UP_6d_rot = _6d_rot[:,:, upper_body_exclude_root].view(*motion.shape[:2], -1)
    UP_joint_velo = joint_velo[:,:, upper_body].view(*motion.shape[:2], -1)
    upper_emb = torch.cat([UP_positions, UP_6d_rot, UP_joint_velo], dim=-1)

    return upper_emb, lower_emb