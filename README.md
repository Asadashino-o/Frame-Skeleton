# Frame-Skeleton
《Frame-Skeleton: A Dual-Stream Network for Action Events Sequence Spotting》
IJCNN2025 paper's origin model 

![Frame-skeleton](./images/model.jpg)

## Dependencies
* [PyTorch](https://pytorch.org/)
PyCharm is recommended.
You can open the Folder as pyCharm project.

## Getting Started
*Please firstly download the video dataset from website(the address is given in the ./data/videos_160/readme)
Place 'all_mp4_file' in the (./data/) directory. 

### Train
* Download the MobileNetV2 pretrained weights from this [repository](https://github.com/tonylins/pytorch-mobilenet-v2) 
and place 'mobilenet_v2.pth.tar' in the root directory. 

* Run train.py

### Evaluate
* Train your own model by following the steps above or download the pre-trained weights (https://pan.baidu.com/s/1-9IvIrOIBCYIxn_U5DQOvQ?pwd=qcpi)
If you download, please put it in the (./models/)directory.

* Run eval.py
 If using the pre-trained weights provided, the PCE should be 0.776.  

### Test your own video by interface
* Follow steps above to download pre-trained weights.(https://pan.baidu.com/s/1-9IvIrOIBCYIxn_U5DQOvQ?pwd=qcpi)&&(https://pan.baidu.com/s/1PjS0kqD5VOvpP2v6MfyCxQ?pwd=p7ra)
 put them in the (./models/)directory.

* Run GUI.py
You will see an interface.Then you can choose random mp4 golf video in your laptop to detect.
