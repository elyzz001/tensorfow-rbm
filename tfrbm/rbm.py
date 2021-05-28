from __future__ import print_function

import tensorflow as tf
#import keras
import numpy as np
import sys
from util import tf_xavier_init
import matplotlib.pyplot as plt

class RBM ():
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate=0.01,
                 momentum=0.95,
                 xavier_const=1.0,
                 err_function='mse',
                 use_tqdm=False,
                 # DEPRECATED:
                 tqdm=None,
                 t = 1):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        self._use_tqdm = use_tqdm
        self._tqdm = None

        if use_tqdm or tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm
        self.temp = t
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])

        self.w = tf.Variable(tf_xavier_init(self.n_visible, self.n_hidden, const=xavier_const), dtype=tf.float32)
        self.visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.update_weights   = None
        self.update_deltas    = None
        self.compute_hidden   = None
        self.compute_visible  = None
        self.compute_visible_from_hidden = None

        self._initialize_vars()

        assert self.update_weights is not None
        assert self.update_deltas is not None
       #
        assert self.compute_hidden1 is not None
        assert self.compute_visible1 is not None
        #assert self.compute_hidden is not None
        #assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None

        if err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(self.x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)
        #else:
            #self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_vars(self):
        pass

    #def get_err(self, batch_x):
        #return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def get_free_energy(self):
        pass

    def transform(self, batch_x):
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x, t):

        #b = batch_x
        #for i in range(iter_num):
        if(t == 0.0):
            #a = self.sess.run(self.compute_visible_real, feed_dict={self.x: batch_x}) #real without binarization
            a = self.sess.run(self.compute_visible1, feed_dict={self.x: batch_x})
            #print("calculation for T = 0")
            #print(" T= ", t)
        else:
            a = self.sess.run(self.compute_visible2, feed_dict={self.x: batch_x})
            #print("calculation for T != 0")
            #print(" T= ",t)
        #b = a.reshape(1,-1)
        #plt.imshow(a.reshape(28, 28))
        #plt.show()
        #print(i)
        return a

    def partial_fit(self, batch_x):
        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.x: batch_x})
       # print("batch x shape",batch_x.shape) #################
    def fit(self,
            data_x,
            n_epoches,
            batch_size=10,
            shuffle=True,
            verbose=True):
        assert n_epoches > 0
        #########
        fname = ["weights_class3kep", "weights_class4kep", "weights_class5kep", "weights_class6kep", "weights_class7kep"
            , "weights_class8kep", "weights_class9kep", "weights_class10kep", "weights_class11kep", "weights_class12kep"
            , "weights_class13kep", "weights_class14kep", "weights_class15kep", "weights_class16kep",
                 "weights_class17kep", "weights_class18kep", "weights_class19kep",
                 "weights_class20kep"]  # ,"5-3","6-3","7-3","8-3","9-3","10-3","11-3","12-3","13-3","14-3","15-3","16-3","17-3","18-3","19-3","20-3"]
        name = ["bbrbm_class3kep", "bbrbm_class4kep", "bbrbm_class5kep", "bbrbm_class6kep", "bbrbm_class7kep"
            , "bbrbm_class8kep", "bbrbm_class9kep", "bbrbm_class10kep", "bbrbm_class11kep", "bbrbm_class12kep"
            , "bbrbm_class13kep", "bbrbm_class14kep", "bbrbm_class15kep", "bbrbm_class16kep", "bbrbm_class17kep"
            , "bbrbm_class18kep", "bbrbm_class19kep", "bbrbm_class20kep"]
        #ep = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ep = np.array([3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,15000,16000,17000,18000,19000,20000])#700, 800, 900, 1000, 1100, 1200,1300,1400, 1500,1600,1700]

        ij = 0
        ##########
        n_data = data_x.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        if shuffle:
            data_x_cpy = data_x.copy()
            inds = np.arange(n_data)
        else:
            data_x_cpy = data_x

        errs = []

        for e in range(n_epoches):
            if verbose and not self._use_tqdm:
                print('Epoch: {:d}'.format(e))

            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            r_batches = range(n_batches)

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

            for b in r_batches:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                self.partial_fit(batch_x)
                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            if verbose:
                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Train error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                    print('Train error: {:.4f}'.format(err_mean))
                    print('')
                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])
            #trying to save the training data for diff ep_num whithout starting over
            if(e == ep[ij]-1 ):
                filename =  fname[ij]
                name1 = name[ij]#'bbrbm_class2ep_inside_rbmpy'
                self.save_weights(filename, name1)
                ij = ij +1

        return errs

    def get_weights(self):
        return self.sess.run(self.w),\
            self.sess.run(self.visible_bias),\
            self.sess.run(self.hidden_bias)

    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        return saver.save(self.sess, filename)

    def set_weights(self, w, visible_bias, hidden_bias):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.visible_bias.assign(visible_bias))
        self.sess.run(self.hidden_bias.assign(hidden_bias))

    def set_temp(self, t):
        self.temp = t
    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        saver.restore(self.sess, filename)
