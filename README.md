# Introduction
This is the source code of our TCSVT 2017 paper "SSDH: Semi-supervised Deep Hashing for Large Scale Image Retrieval", Please cite the following paper if you use our code.

Jian Zhang and Yuxin Peng, "SSDH: Semi-supervised Deep Hashing for Large Scale Image Retrieval", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), doi: 10.1109/TCSVT.2017.2771332, 2017. [[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=20192)

# Dependency
Our code is based on early version of [Caffe](https://github.com/BVLC/caffe), all the dependencies are the same as Caffe.

The proposed SSDH also uses the Pre-trained VGG_CNN_F model, which can be downloaded at [Caffe model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-from-the-bmvc-2014-paper-return-of-the-devil-in-the-details-delving-deep-into-convolutional-nets), download this model and put it in example/semihash/Pre_trained folder.

# Data Preparation
Here we use MIRFlickr dataset for an example, under "data/flickr25k" folder, there are two list, you should resize MIRFlickr dataset according to those two list, so that Caffe can read the image data.

The semitrain_multi_label_h5.list is list of h5 file path. The h5 file contains multi label for the images, which is denoted as a vector, for example, (0,0,0,1,1,0,0,0,0,0,0,0,1,0 ...). For more details, see the code of the multi_label layer.

# Usage

1. Edit Makefile.config to suit your system
2. Compile code: make all -j8
3. Training the model: example/semihash/train_1.sh. You may change train_1.sh to adjust the parameters such as GPU id and model saving location. This code will save the models in models/flickr25k
4. Generate hash codes for testing set: example/semihash/extratfea_flickr25k_12bit.sh. You can adjust script to change hash code saving location GPU id etc. 

For more information, please refer to our [TCSVT paper](http://59.108.48.34/tiki/download_paper.php?fileId=20192).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.
