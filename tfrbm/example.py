
#Hi there, Can you see this 
import numpy as np
import matplotlib.pyplot as plt

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

bbrbm = GBRBM(n_visible=784, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)
errs = bbrbm.fit(mnist_images, n_epoches=10, batch_size=10)
plt.plot(errs)
plt.show()


IMAGE = 1

def show_digit(x):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()

image = mnist_images[IMAGE]
image_rec = bbrbm.reconstruct(image.reshape(1,-1))

show_digit(image)
show_digit(image_rec)


