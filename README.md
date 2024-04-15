# [MMM: Generative Masked Motion Model](https://exitudio.github.io/MMM-page/) (CVPR 2024, Highlight Poster)
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


## Getting Started
### 1. Setup Env
```
conda env create -f environment.yml
conda activate MMM
```
### 2. Get Data
Follow T2M-GPT setup

[2.2. Dependencies](https://github.com/Mael-zys/T2M-GPT?tab=readme-ov-file#22-dependencies)

[2.3. Datasets](https://github.com/Mael-zys/T2M-GPT?tab=readme-ov-file#23-datasets)

### 3. Download Pretrained Models
```
https://drive.google.com/drive/u/1/folders/19qRMMk0mQyA7wyeWU4oZNSFkI6tLxGPN
```
There are 2 folders. Each of which consists of VQVAE and Text-to-Motion transformer models.
1. **text-to-motion**: for text-to-motion and all temporal editing tasks
2. **upper_body_editing**: for upper body editing task.

## Training
#### VQ-VAE
```
python train_vq.py --dataname t2m --exp-name vq_name
```

### Transformer

```
python train_t2m_trans.py --vq-name vq_name --out-dir output/t2m --exp-name trans_name --num-local-layer 1
```
- Make sure the pretrain vqvae in ```output/vq/vq_name/net_last.pth``` <br>
- ```--num-local-layer``` is number of cross attention layer <br>
- support multple gpus ```export CUDA_VISIBLE_DEVICES=0,1,2,3``` <br>
- we use 4 gpus, increasing batch size and iteration to ```--batch-size 512 --total-iter 75000```
- The codebook will be pre-computed and export to ```output/vq/vq_name/codebook``` (It will take a couple minutes.)


### Eval
```
python GPT_eval_multi.py --exp-name eval_name --resume-pth output/vq/vq_name/net_last.pth --resume-trans trans_name
```

## Motion Generation
<summary><b>Text to Motion</b></summary>

```bash
python generate.py --resume-pth '/path/to/vqvae.pth' --resume-trans 'path/to/trans.pth' --text 'the person crouches and walks forward.' --length 156
``````


<summary><b>Editing</b></summary>

Please see detail in
```bash
./edit.ipynb
``````


## License
This code is distributed under an [LICENSE-CC-BY-NC-ND-4.0](LICENSE-CC-BY-NC-ND-4.0.md).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, T2M-GPT, and uses datasets that each have their own respective licenses that must also be followed.
