import tensorflow as tf
import numpy as np
from rbm import RBM
from util import sample_bernoulli

class BBRBMTEMP(RBM):
    def __init__(self, *args, **kwargs):
        RBM.__init__(self, *args, **kwargs)

    def _initialize_vars(self):
        hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        visible_recon_p = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.w)) + self.visible_bias)
        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = tf.matmul(tf.transpose(self.x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        def f(x_old, x_new):
            return self.momentum * x_old +\
            self.learning_rate * x_new * (1 - self.momentum) / tf.to_float(tf.shape(x_new)[0])

        delta_w_new = f(self.delta_w, positive_grad - negative_grad)
        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_p, 0))
        delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        self.update_deltas  = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_w, update_visible_bias, update_hidden_bias]
        print("sel.temp is ", self.temp)
        if(self.temp == 0.0): #temperature zero
            print("i am inside if-sta of t =0")
            compute_hidden_real1 = tf.matmul(self.x, self.w) + self.hidden_bias
            compute_hidden_real1 = tf.where(compute_hidden_real1 < 0.0, tf.zeros_like(compute_hidden_real1),compute_hidden_real1)
            compute_hidden_real1 = tf.where(compute_hidden_real1 > 0.0, tf.ones_like(compute_hidden_real1),compute_hidden_real1)
            #pick zero or one randomly when a = 0

            if(np.random.uniform(0,1) > 0.5):
                pick_x = tf.ones_like(compute_hidden_real1)
            else:
                pick_x = tf.zeros_like(compute_hidden_real1)

            compute_hidden_real = tf.where(compute_hidden_real1 == 0.0, pick_x,compute_hidden_real1)

            #binarize hidden (i think no need to binarize since the above process does that)
                #h_st_bin = tf.math.greater(compute_hidden_real, tf.random.uniform([64]))
                #compute_hidden = tf.cast(h_st_bin, tf.float32)
            self.compute_hidden = compute_hidden_real#compute_hidden########
            #######
            #sigmoid fct for t = 0
            compute_visible_real = tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias
            compute_visible_real = tf.where(compute_visible_real < 0.0, tf.zeros_like(compute_visible_real),
                                            compute_visible_real)
            compute_visible_real = tf.where(compute_visible_real > 0.0, tf.ones_like(compute_visible_real),
                                            compute_visible_real)
            # pick zero or one randomly when a = 0

            if (np.random.uniform(0, 1) > 0.5):
                pick_x = tf.ones_like(compute_visible_real)
            else:
                pick_x = tf.zeros_like(compute_visible_real)

            compute_visible_real = tf.where(compute_visible_real == 0.0, pick_x, compute_visible_real)

            #self.compute_visible_real = compute_visible_real

            #binarize visual
                #v_st_bin = tf.math.greater(compute_visible_real, tf.random.uniform([794]))
                #compute_visible = tf.cast(v_st_bin, tf.float32)
            self.compute_visible = compute_visible_real #compute_visible ########
        else: #temperature other than zero
            print("i am inside else-sta of t not 0")
            #compute hidden units
            compute_hidden_real = tf.nn.sigmoid(tf.math.divide(tf.matmul(self.x, self.w) + self.hidden_bias,self.temp))
            #binarize hidden
            h_st_bin = tf.math.greater(compute_hidden_real, tf.random.uniform([64]))
            compute_hidden = tf.cast(h_st_bin, tf.float32)
            self.compute_hidden = compute_hidden
            #compute the visible units
            compute_visible_real = tf.nn.sigmoid(tf.math.divide(tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias,self.temp))
            # binarize visual
            v_st_bin = tf.math.greater(compute_visible_real, tf.random.uniform([794]))
            compute_visible = tf.cast(v_st_bin, tf.float32)
            self.compute_visible = compute_visible  ########
        ######
        self.compute_visible_from_hidden = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias)