import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding
from exit.utils import cosine_schedule, uniform, top_k, gumbel_sample, top_p
from tqdm import tqdm
from einops import rearrange, repeat
from exit.utils import get_model, generate_src_mask


class PatchUpSampling(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.up_sampling = nn.Linear(dim, 4 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, F, C
        """
        x = self.norm(x)
        x = self.up_sampling(x)
        x0 = x[:, :, 0::4]  
        x1 = x[:, :, 1::4]
        x2 = x[:, :, 2::4]
        x3 = x[:, :, 3::4]
        x = torch.cat([x0, x1, x2, x3], 1)  
        return x

class Decoder_Transformer(nn.Module):
    def __init__(self, 
                code_dim=1024, 
                embed_dim=512, 
                output_dim=263,
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        
        super().__init__()
        self.joint_embed = nn.Linear(code_dim, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.up_sample = PatchUpSampling(embed_dim)
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim),
                            nn.Linear(embed_dim, output_dim))
        self.block_size = block_size
        self.n_head = n_head
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, token_embeddings):
        # token_embeddings = self.tok_emb(idx)
        # B, T = src_mask.shape
        # src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1)

        token_embeddings = token_embeddings.permute(0, 2, 1)
        token_embeddings = self.joint_embed(token_embeddings)
        x = self.pos_embed(token_embeddings)

        # block_step_len = int(len(self.blocks)/3)
        # mask_temp = get_attn_mask(_range=3, _max=x.shape[1]).to(src_mask.device)
        # eye = torch.eye(x.shape[1]).unsqueeze(0).unsqueeze(0).to(src_mask.device).bool()
        # src_mask = src_mask*mask_temp + eye

        for block in self.blocks:
            x = block(x)
        x = self.up_sample(x)

        # mask_2 = mask_1.repeat(1, 1, 2, 2)
        # for block in self.blocks[block_step_len:2*block_step_len]:
        #     x = block(x, mask_2)
        # x = self.up_sample(x)

        # mask_3 = mask_2.repeat(1, 1, 2, 2)
        # for block in self.blocks[2*block_step_len:]:
        #     x = block(x, mask_3)

        x = self.head(x).permute(0, 2, 1)
        return x

# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L342C9-L343C33
class PatchMerging(nn.Module):
    def __init__(self, input_feats, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * input_feats, dim, bias=False)
        self.norm = norm_layer(4 * input_feats)

    def forward(self, x):
        """
        x: B, F, C
        """
        x0 = x[:, 0::4, :]  # B F/2 C
        x1 = x[:, 1::4, :]
        x2 = x[:, 2::4, :]  # B F/2 C
        x3 = x[:, 3::4, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B F/2 2*C
        x = self.norm(x)
        x = self.reduction(x)
        return x

class Encoder_Transformer(nn.Module):
    def __init__(self, 
                input_feats=1024, 
                embed_dim=512, 
                output_dim=263,
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        
        super().__init__()
        self.joint_embed = nn.Linear(input_feats, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        # self.patch_merging1 = PatchMerging(input_feats, embed_dim)
        # self.patch_merging2 = PatchMerging(embed_dim)
        self.weighted_mean_norm = nn.LayerNorm(embed_dim)
        self.weighted_mean = torch.nn.Conv1d(in_channels=block_size, out_channels=1, kernel_size=1)

        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim),
                            nn.Linear(embed_dim, output_dim))
        self.block_size = block_size
        self.n_head = n_head
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, joints):
        # B, T = src_mask.shape
        # src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1)

        joints = joints.permute(0,2,1)
        # token_embeddings = self.joint_embed(joints)

        block_step_len = int(len(self.blocks)/3)

        x = self.joint_embed(joints)
        token_len = int(x.shape[1]/self.block_size)
        _original_shape = list(x.shape)
        x = x.view(x.shape[0]*token_len, self.block_size, -1)

        # mask_temp = get_attn_mask(_range=3, _max=x.shape[1]).to(src_mask.device)
        # eye = torch.eye(x.shape[1]).unsqueeze(0).unsqueeze(0).to(src_mask.device).bool()
        # src_mask = src_mask*mask_temp + eye

        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.weighted_mean_norm(x)
        x = self.weighted_mean(x)
        _original_shape[1] = int(_original_shape[1] / self.block_size)
        x = x.view(*_original_shape)

        # for block in self.blocks[block_step_len:2*block_step_len]:
        #     x = block(x)
        # x = self.patch_merging2(x)

        # for block in self.blocks[2*block_step_len:]:
        #     x = block(x)
        x = self.head(x).permute(0, 2, 1)
        return x

class Text2Motion_Transformer(nn.Module):

    def __init__(self, 
                vqvae,
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                num_local_layer=0, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.n_head = n_head
        self.trans_base = CrossCondTransBase(vqvae, num_vq, embed_dim, clip_dim, block_size, num_layers, num_local_layer, n_head, drop_out_rate, fc_rate)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq


    def get_block_size(self):
        return self.block_size

    def forward(self, *args, type='forward', **kwargs):
        '''type=[forward, sample]'''
        if type=='forward':
            return self.forward_function(*args, **kwargs)
        elif type=='sample':
            return self.sample(*args, **kwargs)
        elif type=='inpaint':
            return self.inpaint(*args, **kwargs)
        else:
            raise ValueError(f'Unknown "{type}" type')
        
    def get_attn_mask(self, src_mask, att_txt=None, txt_mark=None):
        if att_txt is None:
            att_txt = torch.tensor([[True]]*src_mask.shape[0]).to(src_mask.device)
        src_mask = torch.cat([att_txt, src_mask],  dim=1)
        B, T = src_mask.shape
        src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1)
        if txt_mark is not None:
            att_txt_txt = torch.tensor([[True]]*txt_mark.shape[0]).to(txt_mark.device)
            txt_mark = torch.cat([att_txt_txt, txt_mark],  dim=1)
            src_mask[:, :, :, 0] = txt_mark.view(B, 1, T).repeat(1, self.n_head, 1)
        return src_mask

    def forward_function(self, idx_upper, idx_lower, clip_feature, src_mask=None, att_txt=None, txt_mark=None, word_emb=None):
        # MLD:
        # if att_txt is None:
        #     att_txt = torch.tensor([[True]]*src_mask.shape[0]).to(src_mask.device)
        # src_mask = torch.cat([att_txt, src_mask],  dim=1)
        # logits = self.skip_trans(idxs, clip_feature, src_mask)

        # T2M-BD
        if src_mask is not None:
            src_mask = self.get_attn_mask(src_mask, att_txt, txt_mark)
        feat = self.trans_base(idx_upper, idx_lower, clip_feature, src_mask, word_emb)
        logits = self.trans_head(feat, src_mask)

        return logits

    def sample(self, clip_feature, idx_lower, word_emb, m_length=None, if_test=False, rand_pos=False, CFG=-1):
        max_steps = 20
        max_length = 49
        batch_size = clip_feature.shape[0]
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq
        shape = (batch_size, self.block_size - 1)
        topk_filter_thres = .9
        starting_temperature = 1.0
        scores = torch.ones(shape, dtype = torch.float32, device = clip_feature.device)
        
        m_tokens_len = torch.ceil((m_length)/4)
        src_token_mask = generate_src_mask(self.block_size-1, m_tokens_len+1)
        src_token_mask_noend = generate_src_mask(self.block_size-1, m_tokens_len)
        ids = torch.full(shape, mask_id, dtype = torch.long, device = clip_feature.device)
        
        # [TODO] confirm that these 2 lines are not neccessary (repeated below and maybe don't need them at all)
        ids[~src_token_mask] = pad_id # [INFO] replace with pad id
        ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id

        ### PlayGround ####
        # score high = mask
        # m_tokens_len = torch.ceil((m_length)/4)
        # src_token_mask = generate_src_mask(self.block_size-1, m_tokens_len+1)

        # # mock
        # timestep = torch.tensor(.5)
        # rand_mask_prob = cosine_schedule(timestep)
        # scores = torch.arange(self.block_size - 1).repeat(batch_size, 1).cuda()
        # scores[1] = torch.flip(torch.arange(self.block_size - 1), dims=(0,))

        # # iteration
        # num_token_masked = (rand_mask_prob * m_tokens_len).int().clip(min=1)
        # scores[~src_token_mask] = -1e5
        # masked_indices = scores.argsort(dim=-1, descending=True) # This is flipped the order. The highest score is the first in order.
        # masked_indices = masked_indices < num_token_masked.unsqueeze(-1) # So it can filter out by "< num_token_masked". We want to filter the high score as a mask
        # ids[masked_indices] = mask_id
        #########################
        temp = []
        sample_max_steps = torch.round(max_steps/max_length*m_tokens_len) + 1e-8
        for step in range(max_steps):
            timestep = torch.clip(step/(sample_max_steps), max=1)
            rand_mask_prob = cosine_schedule(timestep) # timestep #
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)
            # [INFO] rm no motion frames
            scores[~src_token_mask_noend] = 0
            scores = scores/scores.sum(-1)[:, None] # normalize only unmasked token
            
            # if rand_pos:
            #     sorted_score_indices = scores.multinomial(scores.shape[-1], replacement=False) # stocastic
            # else:
            sorted, sorted_score_indices = scores.sort(descending=True) # deterministic
            
            ids[~src_token_mask] = pad_id # [INFO] replace with pad id
            ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id
            ## [INFO] Replace "mask_id" to "ids" that have highest "num_token_masked" "scores" 
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
            # [INFO] repeat last_id to make it scatter_ the existing last ids.
            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1)-1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index*~select_masked_indices)
            ids.scatter_(-1, sorted_score_indices, mask_id)
            # if torch.isclose(timestep, torch.tensor(0.7647), atol=.01):
            #     print('masked_indices:', ids[0], src_token_mask[0])

            if CFG!=-1:
                # print('ids:', ids.shape, clip_feature.shape, src_token_mask.shape)
                _ids = ids.repeat(2,1)
                _clip_feature = clip_feature.repeat(2,1)
                _src_token_mask = src_token_mask.repeat(2,1)
                att_txt = torch.cat( (torch.ones((batch_size,1), dtype=torch.bool), 
                                      torch.zeros((batch_size,1), dtype=torch.bool) )).to(_ids.device)
                logits = self.forward(_ids, idx_lower, _clip_feature, _src_token_mask, att_txt)[:,1:]
                logits_textcond = logits[:batch_size]
                logits_uncond = logits[batch_size:]
                # logits = (1-CFG)*logits_textcond + CFG*logits_uncond
                logits = (1+CFG)*logits_textcond - CFG*logits_uncond
            else:
                logits = self.forward(ids, idx_lower, clip_feature, src_token_mask, word_emb=word_emb)[:,1:]
            filtered_logits = logits #top_p(logits, .5) # #top_k(logits, topk_filter_thres)
            if rand_pos:
                temperature = 1 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
            else:
                temperature = 0 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
            # pred_ids = filtered_logits.argmax(dim = -1)
            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            is_mask = ids == mask_id
            temp.append(is_mask[:1])
            
            # mid = is_mask[0][:m_tokens_len[0].int()]
            # mid = mid.nonzero(as_tuple=True)[0]
            # print(is_mask[0].sum(), m_tokens_len[0])

            ids = torch.where(
                        is_mask,
                        pred_ids,
                        ids
                    )
            
            # if timestep == 1.:
            #     print(probs_without_temperature.shape)
            probs_without_temperature = logits.softmax(dim = -1)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~is_mask, 0)
        if if_test:
            return ids, temp
        return ids
    
    def inpaint(self, first_tokens, last_tokens, clip_feature=None, inpaint_len=2, rand_pos=False):
        # support only one sample
        assert first_tokens.shape[0] == 1
        assert last_tokens.shape[0] == 1
        max_steps = 20
        max_length = 49
        batch_size = first_tokens.shape[0]
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq
        shape = (batch_size, self.block_size - 1)
        scores = torch.ones(shape, dtype = torch.float32, device = first_tokens.device)
        
        # force add first / last tokens
        first_partition_pos_idx = first_tokens.shape[1]
        second_partition_pos_idx = first_partition_pos_idx + inpaint_len
        end_pos_idx = second_partition_pos_idx + last_tokens.shape[1]

        m_tokens_len = torch.ones(batch_size, device = first_tokens.device)*end_pos_idx

        src_token_mask = generate_src_mask(self.block_size-1, m_tokens_len+1)
        src_token_mask_noend = generate_src_mask(self.block_size-1, m_tokens_len)
        ids = torch.full(shape, mask_id, dtype = torch.long, device = first_tokens.device)
        
        ids[:, :first_partition_pos_idx] = first_tokens
        ids[:, second_partition_pos_idx:end_pos_idx] = last_tokens
        src_token_mask_noend[:, :first_partition_pos_idx] = False
        src_token_mask_noend[:, second_partition_pos_idx:end_pos_idx] = False
        
        # [TODO] confirm that these 2 lines are not neccessary (repeated below and maybe don't need them at all)
        ids[~src_token_mask] = pad_id # [INFO] replace with pad id
        ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id

        temp = []
        sample_max_steps = torch.round(max_steps/max_length*m_tokens_len) + 1e-8

        if clip_feature is None:
            clip_feature = torch.zeros(1, 512).to(first_tokens.device)
            att_txt = torch.zeros((batch_size,1), dtype=torch.bool, device = first_tokens.device)
        else:
            att_txt = torch.ones((batch_size,1), dtype=torch.bool, device = first_tokens.device)

        for step in range(max_steps):
            timestep = torch.clip(step/(sample_max_steps), max=1)
            rand_mask_prob = cosine_schedule(timestep) # timestep #
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)
            # [INFO] rm no motion frames
            scores[~src_token_mask_noend] = 0
            # [INFO] rm begin and end frames
            scores[:, :first_partition_pos_idx] = 0
            scores[:, second_partition_pos_idx:end_pos_idx] = 0
            scores = scores/scores.sum(-1)[:, None] # normalize only unmasked token
            
            sorted, sorted_score_indices = scores.sort(descending=True) # deterministic
            
            ids[~src_token_mask] = pad_id # [INFO] replace with pad id
            ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id
            ## [INFO] Replace "mask_id" to "ids" that have highest "num_token_masked" "scores" 
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
            # [INFO] repeat last_id to make it scatter_ the existing last ids.
            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1)-1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index*~select_masked_indices)
            ids.scatter_(-1, sorted_score_indices, mask_id)

            # [TODO] force replace begin/end tokens b/c the num mask will be more than actual inpainting frames
            ids[:, :first_partition_pos_idx] = first_tokens
            ids[:, second_partition_pos_idx:end_pos_idx] = last_tokens
            
            logits = self.forward(ids, clip_feature, src_token_mask, att_txt)[:,1:]
            filtered_logits = logits #top_k(logits, topk_filter_thres)
            if rand_pos:
                temperature = 1 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
            else:
                temperature = 0 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
            # pred_ids = filtered_logits.argmax(dim = -1)
            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            is_mask = ids == mask_id
            temp.append(is_mask[:1])
            
            ids = torch.where(
                        is_mask,
                        pred_ids,
                        ids
                    )
            
            probs_without_temperature = logits.softmax(dim = -1)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~is_mask, 0)
        return ids

class Attention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, src_mask):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if src_mask is not None:
            att[~src_mask] = float('-inf')
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, src_mask=None):
        x = x + self.attn(self.ln1(x), src_mask)
        x = x + self.mlp(self.ln2(x))
        return x

from models.t2m_trans import Block_crossatt
class CrossCondTransBase(nn.Module):

    def __init__(self, 
                vqvae,
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                num_local_layer = 1,
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.vqvae = vqvae
        # self.tok_emb = nn.Embedding(num_vq + 3, embed_dim).requires_grad_(False) 
        self.learn_tok_emb = nn.Embedding(3, int(self.vqvae.vqvae.code_dim/2))# [INFO] 3 = [end_id, blank_id, mask_id]
        self.to_emb = nn.Linear(self.vqvae.vqvae.code_dim, embed_dim)

        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers-num_local_layer)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.num_local_layer = num_local_layer
        if num_local_layer > 0:
            self.word_emb = nn.Linear(clip_dim, embed_dim)
            self.cross_att = nn.Sequential(*[Block_crossatt(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_local_layer)])
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx_upper, idx_lower, clip_feature, src_mask, word_emb):
        if len(idx_upper) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t = idx_upper.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            learn_idx_upper = idx_upper>=self.vqvae.vqvae.num_code
            learn_idx_lower = idx_lower>=self.vqvae.vqvae.num_code
            
            code_dim = self.vqvae.vqvae.code_dim
            token_embeddings = torch.empty((*idx_upper.shape, code_dim), device=idx_upper.device)
            token_embeddings[..., :int(code_dim/2)][~learn_idx_upper] = self.vqvae.vqvae.quantizer_upper.dequantize(idx_upper[~learn_idx_upper]).requires_grad_(False) 
            token_embeddings[..., :int(code_dim/2)][learn_idx_upper] = self.learn_tok_emb(idx_upper[learn_idx_upper]-self.vqvae.vqvae.num_code)
            token_embeddings[..., int(code_dim/2):][~learn_idx_lower] = self.vqvae.vqvae.quantizer_lower.dequantize(idx_lower[~learn_idx_lower]).requires_grad_(False) 
            token_embeddings[..., int(code_dim/2):][learn_idx_lower] = self.learn_tok_emb(idx_lower[learn_idx_lower]-self.vqvae.vqvae.num_code)
            token_embeddings = self.to_emb(token_embeddings)

            if self.num_local_layer > 0:
                word_emb = self.word_emb(word_emb)
                token_embeddings = self.pos_embed(token_embeddings)
                for module in self.cross_att:
                    token_embeddings = module(token_embeddings, word_emb)
            token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
            
        x = self.pos_embed(token_embeddings)
        for block in self.blocks:
            x = block(x, src_mask)

        return x


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, src_mask):
        for block in self.blocks:
            x = block(x, src_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    


        

