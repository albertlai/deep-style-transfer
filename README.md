# Deep Style Transfer
Tensorflow implementation of the fast feed-forward neural style transfer algo by Johnson et al

Here is an example of styling a photo of San Francisco with Van Gogh's Starry Night
<img src='https://github.com/albertlai/deep-style-transfer/raw/master/data/sf.jpg' height=256/>
<img src='https://github.com/albertlai/deep-style-transfer/raw/master/data/starry.jpg' height=256/>
<img src='https://github.com/albertlai/deep-style-transfer/raw/master/data/styled.jpg' height=256/>

The code is based off [this paper](http://cs.stanford.edu/people/jcjohns/eccv16/) by Johnson et al which in turn builds
off of [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys et al.

This implementation uses [Instance Norm](https://arxiv.org/abs/1607.08022) described by Ulyanov et al and 
[Resize-Convolution](http://distill.pub/2016/deconv-checkerboard/) from Odena et al.

Takes a few hours to train on a P2 instance on AWS and image generation takes a few seconds on a Macbook Pro. Dataset was from MS COCO and uses the VGG19 network for texture and style loss

## Requirements
1. Tensorflow 0.10
2. Pip install
..* scikit-image
..* numpy 1.11 

## Instructions
TODO
