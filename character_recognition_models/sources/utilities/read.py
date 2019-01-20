import sys
import os
import struct
import numpy as np
import tensorflow as tf
import pickle
def read(fname_lbl,fname_img,type = 'd'):
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    if(type == 'd'):

        return lbl, img
    elif(type == 'l'):
        images = []
        for image in img:
            flipped = np.fliplr(image)
            images.append(np.rot90(flipped))
        return lbl, images
    else:
        print("error, type must be l or d")