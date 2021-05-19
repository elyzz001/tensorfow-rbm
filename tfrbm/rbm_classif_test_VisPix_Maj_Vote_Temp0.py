#Hi there, Can you see this
import numpy as np
import tensorflow as tf
from numpy import load
import operator
import tensorflow as tf
import matplotlib.pyplot as plt
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..')) #this line shold always stay above the next line below
from tfrbm import BBRBM, GBRBM
from tensorflow.examples.tutorials.mnist import input_data


#load the mnist data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.train.images
mnist_images1= np.where(mnist_images > 0, 1, 0)

#helper fcts
def pad_with(vector, pad_width, iaxis, kwargs): #https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def show_digit(x,y):
    plt.imshow(x)#,cmap = plt.cm.binary)
    plt.colorbar(mappable=None, cax=None, ax=None)
    plt.title(y)
    plt.locator_params(axis="x", nbins=10)
    plt.locator_params(axis="y", nbins=1)
    plt.show()

#labels pixels
accu = [0]
num_avg = 200
n_data = 1000#mnist_images1.shape[0]
print("accuracy",accu)
t1 = np.zeros(10)

print("minist test size",mnist_images1.shape)
#create the BM
bbrbm = BBRBM(n_visible=794, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)

#load the saved weights
filename = 'weights_class5kep'
name = 'bbrbm_class5kep'
bbrbm.load_weights(filename,name)

#Test the Reconstruction of the RBM
IMAGE = 20#26, 31 works well (which is a 6)
#image = mnist_images1[IMAGE]

mask_a_or =np.ones(784)
mask_c_or =np.zeros(784)
#prepare first mask
mask_bb = mask_a_or
mask_bb = mask_bb.reshape(28,28)
mask_bb = mask_bb[0:22,0:28] #change 16 to lower number to clamp smaller area
mask_b = np.pad(mask_bb, [(0,6), (0,0)], mode='constant')
#show_digit(mask_b.reshape(28, 28), "Mask B")
#prepare second mask
mask_cc = mask_c_or
mask_cc = mask_cc.reshape(28,28)
mask_cc = mask_cc[0:22,0:28]
mask_c = np.pad(mask_cc, [(0, 6), (0, 0)], mode='constant', constant_values=1)
#show_digit(mask_c.reshape(28, 28), "Mask C")
#crop the imag
#crop dimentions
print('size of mask b',mask_b.size)
print('size of mask c', mask_c.size)
#variable to hold the last 10 results of labels
store_recon_vu = np.zeros([num_avg,794])
rec_labels =  np.zeros([1,10])
print('size of store_labels', store_recon_vu.shape)
#reconstruct

#first run
fname = ["1-3","2-3","3-3","4-3","5-3","6-3","7-3","8-3","9-3","10-3","11-3","12-3","13-3","14-3","15-3","16-3","17-3","18-3","19-3","20-3"]
#n_data = mnist_images1.shape[0]
print("number of images", n_data)

for j in range(n_data) :
    #random_image = np.random.uniform(0, 1, 784)
    #image_rec_bin = np.greater(random_image, np.random.uniform(0, 1, 784))
    #random_image = image_rec_bin.astype(int)

    #print(j)
    image = mnist_images1[j]
    #show_digit(image.reshape(28, 28), "Original image")
    #print("image label",mnist.test.labels[j])
    a = image.reshape(28,28)
    c = image.reshape(28, 28)
    #show_digit(image.reshape(28,28))
    # img = a[0:16,0:28] #crop the image
    img = a * mask_b

    img = np.concatenate((t1, img.flatten()), axis=0)
    img_org = img
    #print("shape of of org img", img_org.shape)
    #imga = random_image#imga = img
    #show_digit(img_org[10:794].reshape(28,28),"croped input")
    #reconstruct image for N-MC
    for i in range(400):
        image_rec1 = bbrbm.reconstruct(img.reshape(1,-1),0)
        #print("shape of of rec1",image_rec1.shape)
        image_rec1 = image_rec1.reshape(794, )
        if( i > 400 - num_avg -1):
            store_recon_vu[i - (400 - num_avg)] = image_rec1
            #print("stored labels : ", store_labels)
            #print("index i : ", i)

        #print("new shape of of rec1", image_rec1.shape)
        t1 = image_rec1[0:10]
        rec_backup = image_rec1
        image_rec1 = image_rec1[10:794].reshape(28,28 )
        #print("size ofa", a.size)
        img= img_org + np.concatenate((t1, (image_rec1 * mask_c).flatten()), axis=0)
        #show_digit(image_rec1.reshape(30, 30), "returned image")
        #show_digit(img.reshape(30, 30), "image to be fed")
    #show_digit(img[10:794].reshape(28, 28), "reconstructed image")
    #show_digit(rec_backup[0:10].reshape(1, -1), "reconstructed label")
    #max vote for the correct label
    #print("index i : ", i)
    a = 0
    b = 0
    reconst_err = True
    #for jj in range(store_labels.shape[0]):
        #print("label is ", store_labels[jj])
        #a = a + store_labels[jj][]

    ##add rows of the store_recon_vu array##
    a = np.sum(store_recon_vu, axis=0)
    #print("labels are  ", store_labels)
    #print("a is ", a)
    print("shape of labels is ", store_recon_vu.shape[1])

    ## calculate the majority vote such that if 51 of the iterations is "1" --> Vu = '1' , otherwise Vu = '0'
    for ii in range(store_recon_vu.shape[1]):
        if(a[ii] > num_avg/2):
            a[ii] = 1
            #print("num avrg / 2 ", num_avg/2)
        else:
            a[ii] = 0

    #show_digit(a[10:794].reshape(28, 28), "reconstructed image using VU majority vote")
    ## EXTRACT THE LABELS

    rec_labels = a[0:10]

    for ii in range(10):
        b = b + a[ii]
    if( b > 1): # the network can't decide which one is the number among more than one number ('1' in more than one pixel)
        reconst_err = False #set flag to indicate that
        #print("RBM Confused ")
        #print("a is ", a)
        #print("labels are  ", store_labels)
    else:
        reconst_err = True
    #print("total addition of labels ", a)
    rect_label = np.where(rec_labels == 1)

    #print the result of construction
    #a1 = rec_backup[0:10]
    a2 = mnist.train.labels[j]
    a3 = np.where(a2 == True)

    #print("org label position" , a2)
    #print("recondtructed label (majority vote)", rect_label)

    #if no error occured, increment accuracy if label is correctly reconstructed
    if (reconst_err):
        if(rect_label == a3[0]):
            accu[0] = accu[0] + 1
        #print("accur value" , accu[0])

    #print("org image label", mnist.train.labels[j])
    #print("reconstructed image label", rec_backup[0:10])

    #fig = plt.figure(figsize=(10, 10))

    # setting values to rows and column variables
    """
    rows = 2
    columns = 1
    fig.add_subplot(rows, columns, 2)
    plt.imshow(img[10:794].reshape(28, 28))  # new_image.reshape(29,29)
    plt.title("Reconstructed Image after %i Iterations" %i)
    fig.add_subplot(rows, columns, 1)
    plt.imshow(rec_backup[0:10].reshape(1, -1))
    plt.title("Labels")
    plt.locator_params(axis="x", nbins=10)

    #save figure
    plt.savefig(fname[j], dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
    # plt.show()
    plt.close()
    #plt.show()
    """
print("j aka number of cases",j)
print("accu  ", accu)
accuracy = accu[0]/n_data
print("accuracy  ",accuracy )












