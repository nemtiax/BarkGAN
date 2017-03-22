#Taken from DCGAN code at https://github.com/carpedm20/DCGAN-tensorflow
import scipy.misc
import numpy as np
import math

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def get_image_2(image_path,resize_w):
    im = scipy.misc.imread(image_path).astype(np.float)
    im = scipy.misc.imresize(im,[resize_w,resize_w])
    return im

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        im = scipy.misc.imread(path).astype(np.float)
        # print(im.shape)
        return im

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def save_images(images, size, image_path,resize=-1):
    return imsave(inverse_transform(images), size, image_path,resize)

def save_images_2(images, path):
    image_size = images.shape[1]
    num_images = images.shape[0]
    output_image = np.zeros((image_size * 8,image_size * int(math.ceil(num_images/8.0)),3))
    for i in range(num_images):
        x = i%8
        y = i//8
        image = scipy.misc.toimage(images[i,:,:,:])

        output_image[x*image_size:(x+1)*image_size,y*image_size:(y+1)*image_size,:] = image

    image = scipy.misc.toimage(output_image,cmin=0,cmax=255)
    image.save(path);



def imsave(images, size, path, resize=-1):
    if(resize==-1):
        return scipy.misc.imsave(path, merge(images, size))
    else:
        print("RESIZING OUTPUT!")
        resized = np.array([scipy.misc.imresize(x,[resize,resize]) for x in images[:]])
        #map(, images.tolist()
        return scipy.misc.imsave(path,merge(resized,size))

def inverse_transform(images):
    return (images+1.)/2.
