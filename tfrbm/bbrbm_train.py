#Hi there, Can you see this
import numpy as np
import matplotlib.pyplot as plt
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tfrbm import BBRBM, GBRBM
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.train.images

bbrbm = BBRBM(n_visible=784, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)
errs = bbrbm.fit(mnist_images, n_epoches=100, batch_size=10)
plt.plot(errs)
plt.show()

#save the weights
filename = 'weights'
name = 'bbrbm'
bbrbm.save_weights(filename,name)






