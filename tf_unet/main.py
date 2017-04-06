'''Main function to run

Created on Jul 28, 2016

'''

from __future__ import print_function, division, absolute_import, unicode_literals
import os
import glob
import argparse

from tf_unet import unet, util,image_util

# Use second GPU -- change if you want to use a first one
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default="/home/ajacob6jwu96/snapshots",
                      help='Directory for input data')
parser.add_argument('--complexity', type=int, default = 0, help = 'Complexity of background to use: 0 - all')
parser.add_argument('--output_path', type=str, default = "/home/ajacob6jwu96/codes/athira/ouput", help = 'Output folder')
parser.add_argument('--training_iters', type=int, default = 15)
parser.add_argument('--epochs', type=int, default = 100, help = 'Number of epochs to run for')
parser.add_argument('--restore', type=str, default = False)
parser.add_argument('--layers', type=int, default = 5)
parser.add_argument('--features_root', type=int, default = 64)

FLAGS, unparsed = parser.parse_known_args()

#preparing data loading
data_provider = image_util.ImageDataProvider(FLAGS.data_root, complexity = FLAGS.complexity)

#setup & training
net = unet.Unet(layers=FLAGS.layers, features_root=FLAGS.features_root, channels=3, n_class=2)
trainer = unet.Trainer(net)

path = trainer.train(data_provider, FLAGS.output_path, training_iters=FLAGS.training_iters, epochs=FLAGS.epochs)

#verification
# prediction = net.predict(path, data)
# unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
# img = util.combine_img_prediction(data, label, prediction)
# util.save_image(img, "prediction.jpg")



