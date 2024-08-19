
## Environment Setting
```
conda create -n D3T python=3.8.5
conda activate D3T
# set cuda version to 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install scikit-learn
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
python3 -m pip install -e cvpods

pip install wandb imgaug

# Install some spectial version
pip install numpy==1.20.3
pip install setuptools==59.5.0
pip install Pillow==9.2.0
pip install scikit-learn
pip install scikit-image==0.18.3

```


## Dataset Setting

```
# download Aligned Flir dataset from the link (https://drive.google.com/file/d/1xHDMGl6HJZwtarNWkEV3T4O9X4ZQYz2Y/view)
# unzip to /ailab_mat/dataset/FLIR_ADAS_v2/align
ln -s /ailab_mat/dataset/FLIR_ADAS_v2/align/AnnotatedImages
ln -s /ailab_mat/dataset/FLIR_ADAS_v2/align/Annotations
ln -s /ailab_mat/dataset/FLIR_ADAS_v2/align/JPEGImages

```
[data]
    ├── FLIR_ICIP2020_aligned
          ├── AnnotatedImages
          ├── Annotations
          ├── ImageSets
          └── JPEGImages


## Test
```
cd experiment/flir_rgb2thermal/
CUDA_VISIBLE_DEVICES=1 python3 /SSDe/heeseon/src/D3T/cvpods/tools/test_net.py --dir . --num-gpus 1 MODEL.WEIGHTS /SSDe/heeseon/src/D3T/checkpoint/flir_best.pth


```
