# [MMM: Generative Masked Motion Model](https://exitudio.github.io/MMM-page/) (CVPR 2024, Highlight)
[![arXiv](https://img.shields.io/badge/arXiv-<2312.03596>-<COLOR>.svg)](https://arxiv.org/abs/2312.03596)

The official PyTorch implementation of the paper [**"MMM: Generative Masked Motion Model"**](https://arxiv.org/abs/2312.03596).

Please visit our [**webpage**](https://exitudio.github.io/MMM-page/) for more details.

![teaser_image](https://exitudio.github.io/MMM-page/assets/head.jpg)

If our project is helpful for your research, please consider citing :
``` 
@inproceedings{pinyoanuntapong2024mmm,
  title={MMM: Generative Masked Motion Model}, 
  author={Ekkasit Pinyoanuntapong and Pu Wang and Minwoo Lee and Chen Chen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
}
```
## Update
📢 July/2/24 - Our new paper ([BAMM](https://exitudio.github.io/BAMM-page/)) is accepted at ECCV 2024  <br>
📢 June/10/24 - Update pretrain model with FID. 0.070 (using batchsize 128)<br>
📢 June/8/24 - Interactive demo is live at [huggingface](https://huggingface.co/spaces/samadi10/MMM-Demo) <br>
📢 June/3/24 - Fix generation bugs & add download script & update pretrain model with 2 local layers (better score than reported in the paper)

## Getting Started
### 1. Setup Env
```
conda env create -f environment.yml
conda activate MMM
```

If you have a problem with the conflict, you can install them manually
```
conda create --name MMM
conda activate MMM
conda install plotly tensorboard scipy matplotlib pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install git+https://github.com/openai/CLIP.git einops gdown
pip install --upgrade nbformat
```

### 2. Get Data
### 2.1. Download Glove
```
bash dataset/prepare/download_glove.sh
```


### 2.2. Motion & text feature extractors:

We use the same extractors provided by [t2m](https://github.com/EricGuo5513/text-to-motion) to evaluate our generated motions. Please download the extractors.

```bash
bash dataset/prepare/download_extractor.sh
```

### 2.3. Pre-trained models 

```bash
bash dataset/prepare/download_model.sh
```
### 2.4. Pre-trained models only for upper body editing (optional) 

```bash
bash dataset/prepare/download_model_upperbody.sh
```
<!-- ### 3. Download Pretrained Models
```
https://drive.google.com/drive/u/1/folders/19qRMMk0mQyA7wyeWU4oZNSFkI6tLxGPN
```
There are 2 folders. Each of which consists of VQVAE and Text-to-Motion transformer models.
1. **text-to-motion**: for text-to-motion and all temporal editing tasks
2. **upper_body_editing**: for upper body editing task.

Download and put the pretrained models in `output` folder
`./output/vq/vq_name/net_last.pth` and `./output/t2m/trans_name/net_last.pth` -->



### 2.5. Datasets


We are using two 3D human motion-language dataset: HumanML3D and KIT-ML. For both datasets, you could find the details as well as download link [[here]](https://github.com/EricGuo5513/HumanML3D).   

Take HumanML3D for an example, the file directory should look like this:  
```
./dataset/HumanML3D/
├── new_joint_vecs/
├── texts/
├── Mean.npy # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── Std.npy # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```




## Training
#### VQ-VAE
```
python train_vq.py --dataname t2m --exp-name vq_name
```

### Transformer

```
python train_t2m_trans.py --vq-name vq_name --out-dir output/t2m --exp-name trans_name --num-local-layer 2
```
- Make sure the pretrain vqvae in ```output/vq/vq_name/net_last.pth``` <br>
- ```--num-local-layer``` is number of cross attention layer <br>
- support multple gpus ```export CUDA_VISIBLE_DEVICES=0,1,2,3``` <br>
- we use 4 gpus, increasing batch size and iteration to ```--batch-size 512 --total-iter 75000```
- The codebook will be pre-computed and export to ```output/vq/vq_name/codebook``` (It will take a couple minutes.)


### Eval
```
python GPT_eval_multi.py --exp-name eval_name --resume-pth output/vq/2024-06-03-20-22-07_retrain/net_last.pth --resume-trans output/t2m/2024-06-04-09-29-20_trans_name_b128/net_last.pth --num-local-layer 2
```
The log and tensorboard data will be in ```./output/eval/```
- ```--resume-pth ``` path for vevae
- ```--resume-trans``` path for transformer 
## Motion Generation
<summary><b>Text to Motion</b></summary>

```bash
python generate.py  --resume-pth output/vq/2024-06-03-20-22-07_retrain/net_last.pth --resume-trans output/t2m/2024-06-04-09-29-20_trans_name_b128/net_last.pth --text 'the person crouches and walks forward.' --length 156
``````
The generated html is in ```output``` folder.

<summary><b>Editing</b></summary>
(Please load pretrained model for upper body editing as described in section 2.4) <br>
For better visualization of motion editing, we provide examples of all editing tasks in Jupyter Notebook. Please see the details in:

```bash
./edit.ipynb
``````


## License
This code is distributed under an [LICENSE-CC-BY-NC-ND-4.0](LICENSE-CC-BY-NC-ND-4.0.md).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, T2M-GPT, and uses datasets that each have their own respective licenses that must also be followed.
