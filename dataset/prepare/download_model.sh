echo -e "Downloading pretrain"
mkdir -p ./output/vq/vq_name/
cd ./output/vq/vq_name/
gdown --fuzzy https://drive.google.com/file/d/10ij-p9TR6WVcxVTt5SZSqo22DpVBKGpH/view?usp=drive_link
mv vqvae.pth net_last.pth
cd ../../
mkdir -p ./t2m/trans_name/
cd ./t2m/trans_name/
gdown --fuzzy https://drive.google.com/file/d/1EoKM5a-ki5Zs6MqVsjcb1f5KvoUrAmTF/view?usp=drive_link
mv trans.pth net_last.pth
echo -e "Downloading done!"