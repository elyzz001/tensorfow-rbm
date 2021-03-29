#Hi there, Can you see this
import numpy as np
import matplotlib.pyplot as plt
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tfrbm import BBRBM, GBRBM
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

mnist_images = mnist.train.images
X_test = np.where(mnist_images > 0, 1, 0)

#bbrbm = BBRBM(n_visible=784, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)
#errs = bbrbm.fit(mnist_images, n_epoches=2, batch_size=10)
#plt.plot(errs)
#plt.show()

############################
#Test the Reconstruction of the RBM
IMAGE = 1
image_binarized = X_test[IMAGE]
image = mnist_images[IMAGE]

a = image_binarized.reshape(28,28)
b = image.reshape(28,28)
#img = a[0:16,0:28] #crop the image
#img = cropND(a,(x,y))
#show cropped image
print(image_binarized)
print(image)
plt.imshow(a)
plt.show()
plt.imshow(b)
plt.show()










