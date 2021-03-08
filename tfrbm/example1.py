
#Hi there, Can you see this 
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import keras

import os,sys,inspect
#sys.path.append("..\Users\Abdel\Desktop\Github\tensorfowrbm\tfrbm")

#import sys
#sys.path.insert(0,'C:\Users\Abdel\Desktop\Github\tensorfowrbm\tfrbm\bbrbm')
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tfrbm import BBRBM, GBRBM
#from tfrbm.bbrbm import BBRBM
#from tfrbm.gbrbm import GBRBM
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.train.images

#load the saved Model
errs = load('my_model.npy')

#plot the model
plt.plot(errs)
plt.show()








