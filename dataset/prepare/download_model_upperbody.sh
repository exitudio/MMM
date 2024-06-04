echo -e "Downloading pretrain"
mkdir -p ./output/vq/vq_upperbody/
cd ./output/vq/vq_upperbody/
gdown --fuzzy https://drive.google.com/file/d/1l0_Dva2k7YHL0bI2FTciDJclOCg0aXqb/view?usp=drive_link
mv vqvae.pth net_last.pth
cd ../../
mkdir -p ./t2m/trans_upperbody/
cd ./t2m/trans_upperbody/
gdown --fuzzy https://drive.google.com/file/d/1pi0nAqoAvuoOHpuNS-lMsa_esQG0n-NA/view?usp=drive_link
mv trans.pth net_last.pth
echo -e "Downloading done!"