import os
import skimage
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf

MEAN_VALUES = np.array([123, 117, 104]).reshape((1,1,3))

# synset = [l.strip() for l in open('synset.txt').readlines()]
# returns image of shape [image_h, image_w, 3]
# [height, width, depth]
def load_image(path, image_h, image_w, zoom=False):
    # load image
    img = skimage.io.imread(path)
    if img.ndim < 3:
        img = skimage.color.gray2rgb(img)
    # we crop image from center
    ratio = float(image_h) / image_w
    height = int(img.shape[0])
    width = int(img.shape[1])
    yy = 0
    xx = 0
    if height > width * ratio: #too tall
        yy = int(height - width * ratio) // 2
        height = int(width * ratio)
    else: # too wide
        xx = int(width - height / ratio) // 2
        width = int(height / ratio)
    if zoom:
        yy += int(height / 6)
        xx += int(height / 6)
        height = int(height * 2/ 3)
        width = int(width * 2 / 3)
    crop_img = img[yy: yy + height, xx: xx + width]
    # resize 
    resized_img = skimage.transform.resize(crop_img, (image_h, image_w), preserve_range=True)
    centered_img = resized_img - MEAN_VALUES
    return centered_img

def write_image(path, image, verbose=False):
  img = image[0] + MEAN_VALUES
  if verbose:
      print("%f - %f" % (np.min(img), np.max(img)))
  img = np.clip(img, 0, 255).astype('uint8')
  skimage.io.imsave(path, img)

# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))

def keep_n_newest(directory, n):
    filelist = os.listdir(directory)
    filelist.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    if n < len(filelist):
        for i in xrange(n,len(filelist)):
            os.remove(os.path.join(directory, filelist[i]))

def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
