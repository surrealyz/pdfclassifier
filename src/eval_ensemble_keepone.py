#! /usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from model import Model
import argparse
import pickle
import scipy
from sklearn.metrics import confusion_matrix, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the robustness of insone ensemble model.Every time, deletion is evaluated')
    parser.add_argument('--model_name', type=str, help='Load a previous trained model.', required=True)
    return parser.parse_args()

def classify_either(sess, model, all_vec, splits):
    either_y_pred = []

    # the model.y_input doesn't affect prediction result
    plot_loss, y_pred, pre_softmax, softmax = sess.run([model.plot_loss,model.y_pred, model.pre_softmax, model.y_softmax],
            feed_dict={model.x_input:all_vec,
                model.y_input:np.ones(len(all_vec), dtype=np.float32)
                })

    # get the either_y_pred if any one is malicious
    j = 0
    for cur_cnt in splits:
        cur_mal = False
        for y in y_pred[j:j+cur_cnt]:
            if y == 1:
                cur_mal = True
        j += cur_cnt
        if cur_mal is True:
            either_y_pred.append(1)
        else:
            either_y_pred.append(0)

    return either_y_pred

def classify(sess, model, all_vec):

    # the model.y_input doesn't affect prediction result
    plot_loss, y_pred, pre_softmax, softmax = sess.run([model.plot_loss,model.y_pred, model.pre_softmax, model.y_softmax],
            feed_dict={model.x_input:all_vec,
                model.y_input:np.ones(len(all_vec), dtype=np.float32)
                })
    return y_pred


def classify_intervals_insert(sess, model, x_input, y_input, vectors_all):
    start = 0
    end = 0
    print 'Starting prediction to test VRA...'
    y = y_input.tolist()
    y_input_hat = []
    y_input_ipred = []
    # insertion upper bound
    x_upper = vectors_all
    batch_size = 50
    batch_num = len(x_input)/batch_size
    for i in range(batch_num):
        # no need to print this progress
        #if i % 30 == 0:
        #    print i
        end = start + batch_size
        y_pred, ipred, num_correct, ver_acc = sess.run([model.y_pred, model.interval_pred, model.interval_num_correct, model.verified_accuracy],\
                             feed_dict={model.x_input:x_input[start:end],
                                model.y_input:y_input[start:end],
                                model.upper_input:x_upper[start:end],
                                model.lower_input:x_input[start:end]}

                              )
        # accumulate interval_pred
        y_input_hat += y_pred.tolist()
        y_input_ipred += ipred.tolist()
        start = end

    # remaining, x_input[start:], length < 50. pad the remaining batch to be 50.
    remain_size = len(x_input) - start
    last_x_input = x_input[start:]
    last_y_input = y_input[start:]
    last_x_upper = vectors_all[start:]

    #print 'shape', last_x_input.shape
    last_x_input = np.concatenate((last_x_input, np.array([np.zeros(len(x_input[0])) for i in range(remain_size, 50)])))
    last_y_input = np.concatenate((last_y_input, np.array([0 for i in range(remain_size, 50)])))
    #print 'last_x_input shape', last_x_input.shape
    #print 'last_y_input shape', last_y_input.shape

    last_x_upper = np.concatenate((last_x_upper, np.array([np.ones(len(x_input[0])) for i in range(remain_size, 50)])))
    y_pred, ipred, num_correct, ver_acc = sess.run([model.y_pred, model.interval_pred, model.interval_num_correct, model.verified_accuracy],\
                         feed_dict={model.x_input:last_x_input,
                            model.y_input:last_y_input,
                            model.upper_input:last_x_upper,
                            model.lower_input:last_x_input}
                          )
    y_input_hat += y_pred.tolist()[:remain_size]
    y_input_ipred += ipred.tolist()[:remain_size]

    print('Total intervals evaluated:', len(y_input_ipred))
    return y_input_ipred


def classify_intervals_delete(sess, model, x_input, y_input, vectors_all):
    start = 0
    end = 0
    print 'Starting prediction to test VRA...'
    y = y_input.tolist()
    y_input_hat = []
    y_input_ipred = []
    # insertion upper bound
    x_lower = vectors_all
    batch_size = 50
    batch_num = len(x_input)/batch_size
    for i in range(batch_num):
        # no need to print this progress
        #if i % 30 == 0:
        #    print i
        end = start + batch_size
        y_pred, ipred, num_correct, ver_acc = sess.run([model.y_pred, model.interval_pred, model.interval_num_correct, model.verified_accuracy],\
                             feed_dict={model.x_input:x_input[start:end],
                                model.y_input:y_input[start:end],
                                model.upper_input:x_input[start:end],
                                model.lower_input:x_lower[start:end]}

                              )
        # accumulate interval_pred
        y_input_hat += y_pred.tolist()
        y_input_ipred += ipred.tolist()
        start = end

    # remaining, x_input[start:], length < 50. pad the remaining batch to be 50.
    remain_size = len(x_input) - start
    last_x_input = x_input[start:]
    last_y_input = y_input[start:]
    last_x_lower = vectors_all[start:]

    #print 'shape', last_x_input.shape
    last_x_input = np.concatenate((last_x_input, np.array([np.zeros(len(x_input[0])) for i in range(remain_size, 50)])))
    last_y_input = np.concatenate((last_y_input, np.array([0 for i in range(remain_size, 50)])))
    #print 'last_x_input shape', last_x_input.shape
    #print 'last_y_input shape', last_y_input.shape

    last_x_lower = np.concatenate((last_x_lower, np.array([np.zeros(len(x_input[0])) for i in range(remain_size, 50)])))
    y_pred, ipred, num_correct, ver_acc = sess.run([model.y_pred, model.interval_pred, model.interval_num_correct, model.verified_accuracy],\
                         feed_dict={model.x_input:last_x_input,
                            model.y_input:last_y_input,
                            model.upper_input:last_x_input,
                            model.lower_input:last_x_lower}
                          )
    y_input_hat += y_pred.tolist()[:remain_size]
    y_input_ipred += ipred.tolist()[:remain_size]

    print('Total intervals evaluated:', len(y_input_ipred))
    return y_input_ipred



def main(args):
    # Initialized the model
    model = Model()
    model.tf_interval1(50)

    PATH = '../models/adv_trained/%s.ckpt' % args.model_name
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, PATH)
        print "load model from:", PATH

        feat_trie = pickle.load(open('robustness_spec/feature_spec/pathtrie_filled.pickle', 'rb'))


        print 'Property A and C: testing one subtree deletion vra...'
        # use two deletion vectors to evaluate one deletion
        ### MALICIOUS
        mal_x_input = pickle.load(open('robustness_spec/seed_test_malicious/mutate_keep_one/pickles/x_input.pickle', 'rb'))
        mal_y_input = pickle.load(open('robustness_spec/seed_test_malicious/mutate_keep_one/pickles/y_input.pickle', 'rb'))
        mal_all_vec = pickle.load(open('robustness_spec/seed_test_malicious/mutate_keep_one/pickles/vectors_all.pickle', 'rb'))
        mal_ops_all = pickle.load(open('robustness_spec/seed_test_malicious/mutate_keep_one/pickles/ops_all.pickle', 'rb'))
        sha1_pred = {}
        #vidx_sha1 = {}
        for key, item in mal_ops_all.iteritems():
            sha1, sub1 = item
            if sha1_pred.get(sha1, None) is None:
                sha1_pred[sha1] = {}
            sha1_pred[sha1][sub1] = 0

        #print sha1_pred
        x_lower = np.array([np.zeros(len(mal_x_input[0])) for i in mal_x_input])
        y_pred = classify_intervals_insert(sess, model, x_lower, mal_y_input, mal_all_vec)

        # should still use intervals to get y_pred
        print len(y_pred)
        print sum(y_pred)

        # all sha1_pred[sha1] should be malicious in order to count that sha1.
        for vidx in range(len(y_pred)):
            pred = y_pred[vidx]
            sha1, sub1 = mal_ops_all[vidx]
            if pred == 1:
                sha1_pred[sha1][sub1] = 1

        cnt = 0
        total = 0
        incorrect = 0
        #print sha1_pred
        for sha1, all_sub in sha1_pred.iteritems():
            total += 1
            #print all_sub
            for key, pred in all_sub.iteritems():
                if pred == 0:
                    incorrect += 1
                    break

        print 'total:', total
        print 'incorrect:', incorrect


        print 'Property B, D and E: testing one subtree insertion vra...'
        # Additional test for if any inserted from 0s to 1s are safe
        sha1_pred = {}
        # make an x_upper numpy array, each one has all ones for that subtree.
        x_upper = []
        for key, item in mal_ops_all.iteritems():
            sha1, sub1 = item
            if sha1_pred.get(sha1, None) is None:
                sha1_pred[sha1] = {}
            sha1_pred[sha1][sub1] = 0
            newvec = np.ones(len(mal_x_input[0]))
            min_idx, max_idx = feat_trie[sub1]
            for i in range(min_idx-1, max_idx):
                newvec[i] = 1
            x_upper.append(newvec)

        y_pred = classify_intervals_insert(sess, model, mal_all_vec, mal_y_input, x_upper)

        # should still use intervals to get y_pred
        print len(y_pred)
        print sum(y_pred)

        # all sha1_pred[sha1] should be malicious in order to count that sha1.
        for vidx in range(len(y_pred)):
            pred = y_pred[vidx]
            sha1, sub1 = mal_ops_all[vidx]
            if pred == 1:
                sha1_pred[sha1][sub1] = 1

        cnt = 0
        total = 0
        incorrect = 0
        #print sha1_pred
        for sha1, all_sub in sha1_pred.iteritems():
            total += 1
            #print all_sub
            for key, pred in all_sub.iteritems():
                if pred == 0:
                    incorrect += 1
                    break

        print 'total:', total
        print 'incorrect:', incorrect

        ### HERE
        print 'testing the rest of one insertion vra...'
        # over test 42 zeros to ones. Need the feature range
        for key, obj in feat_trie._root.children.iteritems():
            min_idx, max_idx = feat_trie[key]
            # change seed feature
            upper = np.array([np.zeros(3514)])
            lower = np.array([np.zeros(3514)])
            for i in range(min_idx-1, max_idx):
                upper[0][i] = 1
            # classify each vector and mark corresponding sha1_pred items.
            y_pred = classify_intervals_insert(sess, model, lower, np.ones(1), upper)
            print y_pred




if __name__=='__main__':
    args = parse_args()
    main(args)
