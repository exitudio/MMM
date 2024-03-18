import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from models.t2m_trans import Decoder_Transformer, Encoder_Transformer
from exit.utils import generate_src_mask

class VQVAE_251(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        output_dim = 251 if args.dataname == 'kit' else 263
        self.encoder = Encoder(output_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        
        # Transformer Encoder
        # self.encoder = Encoder_Transformer(
        #     input_feats=output_dim,
        #     embed_dim=512, # 1024
        #     output_dim=512,
        #     block_size=4,
        #     num_layers=6,
        #     n_head=16
        # )

         # Transformer Encoder 4 frames
        # from exit.motiontransformer import MotionTransformerEncoder
        # in_feature = 251 if args.dataname == 'kit' else 263
        # self.encoder2 = MotionTransformerEncoder(in_feature, args.code_dim, num_frames=4, num_layers=2)

        self.decoder = Decoder(output_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        
        # self.decoder = Decoder_Transformer(
        #     code_dim=512,
        #     embed_dim=512, # 1024
        #     output_dim=output_dim,
        #     block_size=49,
        #     num_layers=6,
        #     n_head=8
        # )
        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x):
        
        x_in = self.preprocess(x)
        # Encode
        # _x_in = x_in.reshape( int(x_in.shape[0]*4), x_in.shape[1], 16)
        # x_encoder = self.encoder(_x_in)
        # x_encoder = x_encoder.reshape(x_in.shape[0], -1, int(x_in.shape[2]/4))

        # [Transformer Encoder]
        # _x_in = x_in.reshape( int(x_in.shape[0]*x_in.shape[2]/4), x_in.shape[1], 4)
        # _x_in = _x_in.permute(0,2,1)
        # x_encoder = self.encoder2(_x_in)
        # x_encoder = x_encoder.permute(0,2,1)
        # x_encoder = x_encoder.reshape(x_in.shape[0], -1, int(x_in.shape[2]/4))

        x_encoder = self.encoder(x_in)
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity


    def forward_decoder(self, x):
        # x = x.clone()
        # pad_mask = x >= self.code_dim
        # x[pad_mask] = 0

        x_d = self.quantizer.dequantize(x)
        x_d = x_d.permute(0, 2, 1).contiguous()

        # pad_mask = pad_mask.unsqueeze(1)
        # x_d = x_d * ~pad_mask
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out



class HumanVQVAE(nn.Module):
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
                 norm=None):
        
        super().__init__()
        
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_251(args, nb_code, code_dim, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def forward(self, x, type='full'):
        '''type=[full, encode, decode]'''
        if type=='full':
            x_out, loss, perplexity = self.vqvae(x)
            return x_out, loss, perplexity
        elif type=='encode':
            b, t, c = x.size()
            quants = self.vqvae.encode(x) # (N, T)
            return quants
        elif type=='decode':
            x_out = self.vqvae.forward_decoder(x)
            return x_out
        else:
            raise ValueError(f'Unknown "{type}" type')
        