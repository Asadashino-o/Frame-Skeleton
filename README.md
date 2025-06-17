# Frame-Skeleton
《Frame-Skeleton: A Dual-Stream Network for Action Events Sequence Spotting》
IJCNN2025 paper's origin model 

![Frame-skeleton](./images/model.jpg)

The network architecture of Frame-Skeleton, a two-stream network for action event spotting. Frame-Skeleton splits the video into frames, transforms it into a resized RGB image sequence I and a bone joint sequence K obtained by Pose Extractor, and generates the feature vectors fc and fg by MobileNetV2 and ST-GCN respectively. After passing through a channel attention module to fuse the features, they are processed by a bi-directional LSTM, followed by a fully connected layer with a softmax activation to compute the event probabilities.

The parameter count of the entire model is less than 8 million. While ensuring lightweight, it solves the overfitting problem of small datasets and has a good effect on the key frame detection of single-person movement

## Dependencies

To utilize this tool, ensure the installation of:
- Python 3.8 or later
- [PyTorch](https://pytorch.org/)(preferably with CUDA for GPU support)
- PyCharm is recommended. You can open the Folder as pyCharm project.

## Installation Steps

1. Create a virtual environment for your anaconda or miniconda:
   ```
   conda create -n your_env_name python=3.9
   conda activate your_env_name
   
   ```
   Install the right pytorch version and able to use cuda.
   You can use "nvcc -V" to see your cuda version

2. Clone the repository:
    ```bash
    git clone https://github.com/Asadashino-o/Frame-Skeleton.gi
    cd Frame-Skeleton
    ```
   
3. Install necessary Python packages:
    ```bash
    pip install -r requirements.txt
    conda install opencv -c conda-forge
    ```

4. Clone the Detectron repository:
    ```bash
    git clone https://github.com/facebookresearch/detectron2.git
    ```
   
5. Download the pkl weight file to human-golf/
   Download URL:
   https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl
   
## Getting Started
* Please firstly download the video dataset from website.
You can get the npy files through ./Preprocess/generate_npy.py. 

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
