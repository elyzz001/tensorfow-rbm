#Hi there, Can you see this
import numpy as np
import matplotlib.pyplot as plt
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tfrbm import BBRBM, GBRBM
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.train.images
#X_test = np.where(mnist_images > 0, 1, 0) #binrize the pixels

bbrbm = BBRBM(n_visible=784, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)
errs = bbrbm.fit(mnist_images, n_epoches=1000, batch_size=10)
plt.plot(errs)
plt.show()

############################
#Test the Reconstruction of the RBM
IMAGE = 1
image = mnist_images[IMAGE]
#crop the imag
#crop dimentions
x = 6
y = 6
a = image.reshape(28,28)
img = a[0:16,0:28] #crop the image
#img = cropND(a,(x,y))
#show cropped image
#rint(img)
plt.imshow(a)
plt.show()
plt.imshow(img)
plt.show()
#pad the image to make it 780 before feeding it to the BM
imge = np.pad(img, [(6, ), (0, )], mode='constant')
#print(imge)
plt.imshow(imge)
plt.show()
#reconstruct
image_rec = bbrbm.reconstruct(imge.reshape(1,-1))
#show_digit(image)
#show_digit(image_rec)
#plot reconstructed image
plt.imshow(image_rec.reshape(28,28))
plt.show()

############################
#save the weights
filename = 'weights'
name = 'bbrbm'
bbrbm.save_weights(filename,name)








