# download backbones: resnet101 and resnet152 (only if you want training by yourself)
wget http://guanghan.info/download/Data/LightTrack/weights/backbones.zip
unzip backbones.zip

# download weights for the detection module
wget http://guanghan.info/download/Data/LightTrack/weights/YOLOv3.zip
unzip YOLOv3.zip

# download weights for various pose estimators
wget http://guanghan.info/download/Data/LightTrack/weights/mobile-deconv.zip
unzip mobile-deconv.zip
wget http://guanghan.info/download/Data/LightTrack/weights/CPN101.zip
unzip CPN101.zip
wget http://guanghan.info/download/Data/LightTrack/weights/MSRA152.zip
unzip MSRA152.zip

# download weights for the SGCN module
wget http://guanghan.info/download/Data/LightTrack/weights/GCN.zip
unzip GCN.zip
