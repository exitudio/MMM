def get_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model

import numpy as np
import torch
from utils.motion_process import recover_from_ric
import copy
import plotly.graph_objects as go
import shutil
import datetime
import os
import math

kit_bone = [[0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [0, 16], [16, 17], [17, 18], [18, 19], [19, 20], [0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [6, 7], [3, 8], [8, 9], [9, 10]]
t2m_bone = [[0,2], [2,5],[5,8],[8,11],
            [0,1],[1,4],[4,7],[7,10],
            [0,3],[3,6],[6,9],[9,12],[12,15],
            [9,14],[14,17],[17,19],[19,21],
            [9,13],[13,16],[16,18],[18,20]]
kit_kit_bone = kit_bone + (np.array(kit_bone)+21).tolist()
t2m_t2m_bone = t2m_bone + (np.array(t2m_bone)+22).tolist()

def axis_standard(skeleton):
    skeleton = skeleton.copy()
#     skeleton = -skeleton
    # skeleton[:, :, 0] *= -1
    # xyz => zxy
    skeleton[..., [1, 2]] = skeleton[..., [2, 1]]
    skeleton[..., [0, 1]] = skeleton[..., [1, 0]]
    return skeleton

def visualize_2motions(motion1, std, mean, dataset_name, length, motion2=None, save_path=None):
    motion1 = motion1 * std + mean
    if motion2 is not None:
        motion2 = motion2 * std + mean
    if dataset_name == 'kit':
        first_total_standard = 60
        bone_link = kit_bone
        if motion2 is not None:
            bone_link = kit_kit_bone
        joints_num = 21
        scale = 1/1000
    else:
        first_total_standard = 63
        bone_link = t2m_bone
        if motion2 is not None:
            bone_link = t2m_t2m_bone
        joints_num = 22
        scale = 1#/1000
    joint1 = recover_from_ric(torch.from_numpy(motion1).float(), joints_num).numpy()
    if motion2 is not None:
        joint2 = recover_from_ric(torch.from_numpy(motion2).float(), joints_num).numpy()
        joint_original_forward = np.concatenate((joint1, joint2), axis=1)
    else:
        joint_original_forward = joint1
    animate3d(joint_original_forward[:length]*scale, 
              BONE_LINK=bone_link, 
              first_total_standard=first_total_standard, 
              save_path=save_path) # 'init.html'
    
def animate3d(skeleton, BONE_LINK=t2m_bone, first_total_standard=-1, root_path=None, root_path2=None, save_path=None, axis_standard=axis_standard, axis_visible=True):
    # [animation] https://community.plotly.com/t/3d-scatter-animation/46368/6
    
    SHIFT_SCALE = 0
    START_FRAME = 0
    NUM_FRAMES = skeleton.shape[0]
    skeleton = skeleton[START_FRAME:NUM_FRAMES+START_FRAME]
    skeleton = axis_standard(skeleton)
    if BONE_LINK is not None:
        # ground truth
        bone_ids = np.array(BONE_LINK)
        _from = skeleton[:, bone_ids[:, 0]]
        _to = skeleton[:, bone_ids[:, 1]]
        # [f 3(from,to,none) d]
        bones = np.empty(
            (_from.shape[0], 3*_from.shape[1], 3), dtype=_from.dtype)
        bones[:, 0::3] = _from
        bones[:, 1::3] = _to
        bones[:, 2::3] = np.full_like(_to, None)
        display_points = bones
        mode = 'lines+markers'
    else:
        display_points = skeleton
        mode = 'markers'
    # follow this thread: https://community.plotly.com/t/3d-scatter-animation/46368/6
    fig = go.Figure(
        data=go.Scatter3d(  x=display_points[0, :first_total_standard, 0], 
                            y=display_points[0, :first_total_standard, 1],
                            z=display_points[0, :first_total_standard, 2], 
                            name='Nodes0',
                            mode=mode, 
                            marker=dict(size=3, color='blue',)), 
                            layout=go.Layout(
                                scene=dict(aspectmode='data', 
                                camera=dict(eye=dict(x=3, y=0, z=0.1)))
                                )
                            )
    if first_total_standard != -1:
        fig.add_traces(data=go.Scatter3d(  
                                x=display_points[0, first_total_standard:, 0], 
                                y=display_points[0, first_total_standard:, 1],
                                z=display_points[0, first_total_standard:, 2], 
                                name='Nodes1',
                                mode=mode, 
                                marker=dict(size=3, color='red',)))

    if root_path is not None:
        root_path = axis_standard(root_path)
        fig.add_traces(data=go.Scatter3d(  
                                    x=root_path[:, 0], 
                                    y=root_path[:, 1],
                                    z=root_path[:, 2], 
                                    name='root_path',
                                    mode=mode, 
                                    marker=dict(size=2, color='green',)))
    if root_path2 is not None:
        root_path2 = axis_standard(root_path2)
        fig.add_traces(data=go.Scatter3d(  
                                    x=root_path2[:, 0], 
                                    y=root_path2[:, 1],
                                    z=root_path2[:, 2], 
                                    name='root_path2',
                                    mode=mode, 
                                    marker=dict(size=2, color='red',)))

    frames = []
    # frames.append({'data':copy.deepcopy(fig['data']),'name':f'frame{0}'})

    def update_trace(k):
        fig.update_traces(x=display_points[k, :first_total_standard, 0],
            y=display_points[k, :first_total_standard, 1],
            z=display_points[k, :first_total_standard, 2],
            mode=mode,
            marker=dict(size=3, ),
            # traces=[0],
            selector = ({'name':'Nodes0'}))
        if first_total_standard != -1:
            fig.update_traces(x=display_points[k, first_total_standard:, 0],
                y=display_points[k, first_total_standard:, 1],
                z=display_points[k, first_total_standard:, 2],
                mode=mode,
                marker=dict(size=3, ),
                # traces=[0],
                selector = ({'name':'Nodes1'}))

    for k in range(0, len(display_points)):
        update_trace(k)
        frames.append({'data':copy.deepcopy(fig['data']),'name':f'frame{k}'})
    update_trace(0)

    # frames = [go.Frame(data=[go.Scatter3d(
    #     x=display_points[k, :, 0],
    #     y=display_points[k, :, 1],
    #     z=display_points[k, :, 2],
    #     mode=mode,
    #     marker=dict(size=3, ))],
    #     traces=[0],
    #     name=f'frame{k}'
    # )for k in range(len(display_points))]
    
    
    
    fig.update(frames=frames)

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {"pad": {"b": 10, "t": 60},
         "len": 0.9,
         "x": 0.1,
         "y": 0,

         "steps": [
            {"args": [[f.name], frame_args(0)],
             "label": str(k),
             "method": "animate",
             } for k, f in enumerate(fig.frames)
        ]
        }
    ]

    fig.update_layout(
        updatemenus=[{"buttons": [
            {
                "args": [None, frame_args(1000/25)],
                "label": "Play",
                "method": "animate",
            },
            {
                "args": [[None], frame_args(0)],
                "label": "Pause",
                "method": "animate",
            }],

            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": 0.1,
            "y": 0,
        }
        ],
        sliders=sliders
    )
    range_x, aspect_x = get_range(skeleton, 0)
    range_y, aspect_y = get_range(skeleton, 1)
    range_z, aspect_z = get_range(skeleton, 2)

    fig.update_layout(scene=dict(xaxis=dict(range=range_x, visible=axis_visible),
                                 yaxis=dict(range=range_y, visible=axis_visible),
                                 zaxis=dict(range=range_z, visible=axis_visible)
                                 ),
                      scene_aspectmode='manual',
                      scene_aspectratio=dict(
                          x=aspect_x, y=aspect_y, z=aspect_z)
                      )

    fig.update_layout(sliders=sliders)
    fig.show()
    if save_path is not None:
        fig.write_html(save_path, auto_open=False, include_plotlyjs='cdn', full_html=False)

def get_range(skeleton, index):
    _min, _max = skeleton[:, :, index].min(), skeleton[:, :, index].max()
    return [_min, _max], _max-_min

# [INFO] from http://juditacs.github.io/2018/12/27/masked-attention.html
def generate_src_mask(T, length):
    B = len(length)
    mask = torch.arange(T).repeat(B, 1).to(length.device) < length.unsqueeze(-1)
    return mask

def copyComplete(source, target):
    '''https://stackoverflow.com/questions/19787348/copy-file-keep-permissions-and-owner'''
    # copy content, stat-info (mode too), timestamps...
    if os.path.isfile(source):
        shutil.copy2(source, target)
    else:
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
    # copy owner and group
    st = os.stat(source)
    os.chown(target, st.st_uid, st.st_gid)

data_permission = os.access('/data/epinyoan', os.R_OK | os.W_OK | os.X_OK)
base_dir = '/data' if data_permission else '/home'
def init_save_folder(args, copysource=True):
    import glob
    global base_dir
    if args.exp_name != 'TEMP':
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.out_dir = f"./{args.out_dir}/{date}_{args.exp_name}/"
        save_source = f'{args.out_dir}source/'
        os.makedirs(save_source, mode=os.umask(0), exist_ok=False)
    else:
        args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')

def uniform(shape, device = None):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    # [INFO] select top 10% samples of last index by fill value to the rest as -inf
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

# https://github.com/lucidrains/DALLE-pytorch/issues/318
# https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
from torch.nn import functional as F
def top_p(logits, thres = 0.1):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > (1 - thres)
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

    logits[indices_to_remove] = float('-inf')
    return logits