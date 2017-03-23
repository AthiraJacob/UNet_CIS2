'''Main function to run

Created on Jul 28, 2016

'''

from __future__ import print_function, division, absolute_import, unicode_literals
import os
import glob
import click

from tf_unet import unet, util,image_util

@click.command()
@click.option('--data_root', default="/home/ajacob6jwu96/snapshots/CIS2")
@click.option('--complexity', default= "all"
@click.option('--output_path', default="/home/ajacob6jwu96/codes/athira/ouput")
@click.option('--training_iters', default=32)
@click.option('--epochs', default=100)
@click.option('--restore', default=False)
@click.option('--layers', default=5)
@click.option('--features_root', default=64)

#preparing data loading
data_provider = image_util.ImageDataProvider(data_root, complexity)

#setup & training
net = unet.Unet(layers=layers, features_root=features_root, channels=3, n_class=2)
trainer = unet.Trainer(net)

path = trainer.train(data_provider, output_path, training_iters=training_iters, epochs=epochs)

#verification
...
prediction = net.predict(path, data)
unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
img = util.combine_img_prediction(data, label, prediction)
util.save_image(img, "prediction.jpg")


