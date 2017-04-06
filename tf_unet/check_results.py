from __future__ import print_function, division, absolute_import, unicode_literals
import os
import glob
import argparse

from tf_unet import unet, util,image_util
import os
import shutil
import numpy as np
import cv2

from collections import OrderedDict
import logging

import tensorflow as tf

from tf_unet import util
from tf_unet.layers import (weight_variable, weight_variable_devonc, bias_variable, 
                            conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax_2,
                            cross_entropy)

FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default="/home/ajacob6jwu96/codes/athira/data_trial",
                      help='Directory for input data')
parser.add_argument('--model_path', type=str, default="/home/ajacob6jwu96/codes/athira/ouput",
                      help='Directory for network')

FLAGS, unparsed = parser.parse_known_args()

fname = '/home/ajacob6jwu96/codes/athira/tf_unet/tf_unet/prediction/img.png'
img = np.squeeze(cv2.imread(fname))
  
x_test = img.reshape([1,img.shape[0],img.shape[1],img.shape[2]])

n_class = 2

net = unet.Unet(layers=5, features_root=64, channels=3, n_class=n_class)

with tf.Session() as sess:

	ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
	if ckpt and ckpt.model_checkpoint_path:
		net.restore(sess, ckpt.model_checkpoint_path)
	y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], n_class))
	prediction = sess.run(net.predicter, feed_dict={net.x: x_test, 
                                                             net.y: y_dummy, 
                                                             net.keep_prob: 1.})

print(prediction.shape)
cv2.imwrite('input.png',img)
cv2.imwrite('output.png',prediction[0,:,:,0]*255)

