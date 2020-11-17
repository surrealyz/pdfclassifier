"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Model(object):
    def __init__(self):
        self.x_input = tf.placeholder(tf.float32, shape = [None, 3514])
        self.y_input = tf.placeholder(tf.int64, shape = [None])

        h_fc1 = tf.layers.dense(self.x_input, 200, activation=tf.nn.relu, name="fc1")
        h_fc2 = tf.layers.dense(h_fc1, 200, activation=tf.nn.relu, name="fc2")

        self.pre_softmax = tf.layers.dense(h_fc2, 2, activation=None, name="pre_softmax")
        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y_input, logits=self.pre_softmax)
        
        self.y_softmax = tf.nn.softmax(self.pre_softmax)
        
        self.plot_loss = y_xent
        self.xent = tf.reduce_sum(y_xent)

        self.y_pred = tf.argmax(self.pre_softmax, 1)

        self.correct_prediction = tf.equal(self.y_pred, self.y_input)

        self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.cast(self.num_correct, tf.float32) / tf.cast(tf.shape(self.y_input)[0], tf.float32)

        _, self.accuracy_op =\
                        tf.metrics.accuracy(labels=self.y_input,\
                        predictions=self.y_pred)
        
        _, self.false_positive_op = tf.metrics.false_positives(self.y_input, self.y_pred) 


        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        label_mask = tf.one_hot(self.y_input,
                                2,
                                on_value=1.0,
                                off_value=0.0,
                                dtype=tf.float32)
       
    
    def tf_propagate1(self, center, error_range, equation, error_row, bias_row):
        #(784,512)*(100,784,1)->(100,784,512)->(100,512)
        l1 = tf.reduce_sum(tf.abs(equation)*tf.expand_dims(error_range, axis=-1), axis=1)
        upper = center+l1+bias_row+tf.reduce_sum(error_row*\
                        tf.cast(tf.greater(error_row,0), tf.float32),axis=1)

        lower = center-l1+bias_row+tf.reduce_sum(error_row*\
                        tf.cast(tf.less(error_row,0), tf.float32), axis=1)
        
        appr_condition = tf.cast(tf.logical_and(tf.less(lower,0),tf.greater(upper,0)), tf.float32)
        mask = appr_condition*((upper)/(upper-lower+tf.math.exp(-10.0)))
        mask = mask + 1 - appr_condition
        mask = mask*tf.cast(tf.greater(upper, 0), tf.float32)

        bias_row = bias_row*mask
        center = center*mask

        #mask=(100,1,512)
        mask = tf.expand_dims(mask,axis=1)
        #(784,512)*(100,1,512)
        equation = equation*mask
        #(1,512)*(100,1,512)

        I = tf.eye(tf.shape(mask)[-1], dtype=tf.float32)
        
        error_row = tf.concat([error_row,\
                                tf.expand_dims(tf.negative(lower), axis=1)*\
                                I*tf.expand_dims(appr_condition, axis=1)], axis=1)
        
        error_row = error_row*mask

        return upper, lower, center, equation, error_row, bias_row


    def tf_interval1(self, batch_size):
        self.upper_input = tf.placeholder(tf.float32, shape = [None, 3514])
        self.lower_input = tf.placeholder(tf.float32, shape = [None, 3514])
        upper_input = self.upper_input
        lower_input = self.lower_input

        error_range = (upper_input-lower_input)/2.0
        center = (lower_input+upper_input)/2
        m = 3514
        equation = tf.eye(m, dtype=tf.float32)
        bias_row = tf.zeros([1,m], dtype=tf.float32)
        error_row = tf.zeros([tf.shape(self.x_input)[0],1,m], dtype=tf.float32)
        
        center = tf.layers.dense(inputs=center, units=200, activation=None,\
                                                name="fc1", reuse=True, use_bias = False)
        equation = tf.layers.dense(inputs=equation, units=200, activation=None,\
                                                name="fc1", reuse=True, use_bias = False)
        bias_row = tf.layers.dense(inputs=bias_row, units=200, activation=None,\
                                                name="fc1", reuse=True)
        error_row = tf.layers.dense(inputs=error_row, units=200, activation=None,\
                                                name="fc1", reuse=True, use_bias = False)
        
        upper, lower, center, equation, error_row, bias_row=\
                                self.tf_propagate1(center, error_range, equation, error_row, bias_row)
        
        
        center = tf.layers.dense(inputs=center, units=200, activation=None,\
                                                name="fc2", reuse=True, use_bias = False)
        equation = tf.layers.dense(inputs=equation, units=200, activation=None,\
                                                name="fc2", reuse=True, use_bias = False)
        bias_row = tf.layers.dense(inputs=bias_row, units=200, activation=None,\
                                                name="fc2", reuse=True)
        error_row = tf.layers.dense(inputs=error_row, units=200, activation=None,\
                                                name="fc2", reuse=True, use_bias = False)
        
        upper, lower, center, equation, error_row, bias_row =\
                                self.tf_propagate1(center, error_range, equation, error_row, bias_row)
        
        equation2 = equation

        center = tf.layers.dense(inputs=center, units=2, activation=None,\
                                                name="pre_softmax", reuse=True, use_bias = False)
        equation = tf.layers.dense(inputs=equation, units=2, activation=None,\
                                                name="pre_softmax", reuse=True, use_bias = False)
        bias_row = tf.layers.dense(inputs=bias_row, units=2, activation=None,\
                                                name="pre_softmax", reuse=True)
        error_row = tf.layers.dense(inputs=error_row, units=2, activation=None,\
                                                name="pre_softmax", reuse=True, use_bias = False)

        # normalized the output

        center_t = center[0:1, self.y_input[0]]
        equation_t = equation[0:1,:,self.y_input[0]]
        bias_row_t = bias_row[0:1, self.y_input[0]]
        error_row_t = error_row[0:1, :, self.y_input[0]]
        
        for i in range(1, batch_size):
            center_t = tf.concat([center_t, center[i:i+1, self.y_input[i]]], axis=0)
            equation_t = tf.concat([equation_t, equation[i:i+1,:, self.y_input[i]]], axis=0)
            bias_row_t = tf.concat([bias_row_t, bias_row[i:i+1, self.y_input[i]]], axis=0)
            error_row_t = tf.concat([error_row_t, error_row[i:i+1,:, self.y_input[i]]], axis=0)

        center = center-tf.expand_dims(center_t, axis=-1)
        equation = equation-tf.expand_dims(equation_t, axis=-1)
        bias_row = bias_row-tf.expand_dims(bias_row_t, axis=-1)
        error_row = error_row-tf.expand_dims(error_row_t, axis=-1)

        upper, lower, center, equation, error_row, bias_row =\
                                self.tf_propagate1(center, error_range, equation, error_row, bias_row)

        self.equation = equation
        self.error_row = error_row
        self.bias_row = bias_row

       
        d_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y_input, logits=upper)
        self.interval_xent = tf.reduce_sum(d_xent)

        # prediction based on upper bound of propagation
        self.interval_pred = tf.argmax(upper, 1)

        correct_prediction = tf.equal(self.interval_pred, self.y_input)

        self.interval_num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
        self.verified_accuracy = tf.cast(self.interval_num_correct, tf.float32) / tf.cast(tf.shape(self.y_input)[0], tf.float32)

        _, self.verified_accuracy_op =\
                        tf.metrics.accuracy(labels=self.y_input,\
                        predictions=self.interval_pred)


        return



