# Deep Style Transfer
Tensorflow implementation of the fast feed-forward neural style transfer network by Johnson et al.

Here is an example of styling a photo of San Francisco with Van Gogh's Starry Night
<img src='https://github.com/albertlai/deep-style-transfer/raw/master/data/sf.jpg' height=256/>
<img src='https://github.com/albertlai/deep-style-transfer/raw/master/data/starry.jpg' height=256/>
<img src='https://github.com/albertlai/deep-style-transfer/raw/master/data/styled.jpg' height=256/>

The code is based off [this paper](http://cs.stanford.edu/people/jcjohns/eccv16/) by Johnson et al which in turn builds
off of [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys et al.

This implementation uses [Instance Norm](https://arxiv.org/abs/1607.08022) described by Ulyanov et al and 
[Resize-Convolution](http://distill.pub/2016/deconv-checkerboard/) from Odena et al.

Takes a few hours to train on a P2 instance on AWS and image generation takes a few seconds on a Macbook Pro. Training image dataset was from MS COCO validation set and uses the VGG19 network for texture and style loss

## Requirements
1. Tensorflow 0.10
2. pip install:
  * scikit-image
  * numpy 1.11 

## Instructions for Processing
1. Go to the project root (there is a pretrained model in the /data directory)
2. Run:
```
$ python style.py --input=path_to_image.jpg --output=your_output_file.jpg
```
## Intructions for Training
1. Download [VGG19 weights](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) as vgg19.npy
2. Download [MS COCO validation dataset](http://mscoco.org/dataset/#download) - the training script defaults to look for images in a directory named input_images, though you can add whatever directory you want as a command line argument
3. Run:
```
$ python train.py --data_dir=/path/to/ms_coco --texture=path/to/source_image.jpg
```

