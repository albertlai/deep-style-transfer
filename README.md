# deep-style-transfer
Tensorflow implementation of the fast feed-forward neural style transfer algo by Johnson et al

This code is based off [this paper](http://cs.stanford.edu/people/jcjohns/eccv16/) by Johnson et al which in turn builds
off of [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys et al.

This implementation uses [Instance Norm](https://arxiv.org/abs/1607.08022) described by Ulyanov et al and 
[Resize-Convolution](http://distill.pub/2016/deconv-checkerboard/) from Odena et al.

Takes a few hours to train on a P2 instance on AWS and image generation takes a few seconds on a Macbook Pro.
