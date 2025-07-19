 
# Install
## 1. Clone (or download) the source code 
```
https://github.com/taco-group/AirV2X-Perception.git
cd AirV2X-Perception
```
 
## 2. Create conda environment and set up the base dependencies
```
conda create --name airv2x python=3.7 cmake=3.22.1 setuptools=58.0
conda activate airv2x
conda install cudnn -c conda-forge
conda install boost
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## 3. Install spconv (Support both 1.2.1 and 2.x)

### *(Notice): Make sure *libboost-all-dev* is installed in your linux system before installing *spconv*. If not:
```
sudo apt-get install libboost-all-dev
```

## Install 2.x
```
pip install spconv-cu113
```

## 4. Install pypcd
```
git clone https://github.com/klintan/pypcd.git
cd pypcd
pip install python-lzf
python setup.py install
cd ..
```

## 5. Install airv2x
```
# install requirements
pip install -r requirements.txt
python setup.py develop

# Bbx IOU cuda version compile
python opencood/utils/setup.py build_ext --inplace

# FPVRCNN's iou_loss dependency
python opencood/pcdet_utils/setup.py build_ext --inplace
```

### if there is a problem about cv2:
```
# module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'
pip install "opencv-python-headless<4.3"
```



