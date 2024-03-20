# [MMM: Generative Masked Motion Model](https://exitudio.github.io/MMM-page/)
[![arXiv](https://img.shields.io/badge/arXiv-<2312.03596>-<COLOR>.svg)](https://arxiv.org/abs/2312.03596)

The official PyTorch implementation of the paper [**"MMM: Generative Masked Motion Model"**](https://arxiv.org/abs/2312.03596).

Please visit our [**webpage**](https://exitudio.github.io/MMM-page/) for more details.

![teaser_image](https://exitudio.github.io/MMM-page/assets/head.jpg)


## Getting Started
### 1. Setup Env
```
conda env create -f environment.yml
conda activate MMM
```
### 2. Get Data
### 3. Download Pretrained Models
```
https://drive.google.com/drive/u/1/folders/19qRMMk0mQyA7wyeWU4oZNSFkI6tLxGPN
```

### 4. training
#### vq
```
python train_vq.py --dataname t2m --exp-name vq_name
```

## Motion Generation
<details>
  <summary><b>Text to Motion</b></summary>
  
  ```bash
  python generate.py --resume-pth '/path/to/vqvae.pth' --resume-trans 'path/to/trans.pth' --text 'the person crouches and walks forward.' --length 156
  ``````
</details>

<!-- <details>
  <summary><b>Motion Temporal Editing</b></summary>
</details>

<details>
  <summary><b>Upper Body Editing</b></summary>
</details>

<details>
  <summary><b>Long Range Motion Generation</b></summary>
</details> -->
