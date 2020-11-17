#! /usr/bin/env python
import os
import numpy as np
from sklearn import datasets
import tensorflow as tf
from model import Model
import argparse
import pickle
import scipy
from sklearn.metrics import confusion_matrix, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(description='Use a trained model to make an ensemble.')
    parser.add_argument('--model_name', type=str, help='Load a previous trained model.', required=True)
    return parser.parse_args()

def eval(y, y_p):
    try:
        tn, fp, fn, tp = confusion_matrix(y, y_p).ravel()
        acc = (tp+tn)/float(tp+tn+fp+fn)
        fpr = fp/float(fp+tn)
        return acc, fpr
    except ValueError:
        return accuracy_score(y, y_p), None

def main(args):
    # Initialized the model
    model = Model()

    PATH = '../models/adv_trained/%s.ckpt' % args.model_name
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, PATH)
        print "load model from:", PATH


        # try delete all the possible one subtree, and do an ensemble.
        # I need the indices in the feature space? that's for online

        ### MALICIOUS
        mal_all_vec = pickle.load(open('robustness_spec/seed_test_malicious/mutate_delete_one/pickles/vectors_all.pickle', 'rb'))
        mal_splits = pickle.load(open('robustness_spec/seed_test_malicious/mutate_delete_one/pickles/splits.pickle', 'rb'))
        #mal_y_input = pickle.load(open('robustness_spec/mutate_delete_one_test/pickles/y_input.pickle', 'rb'))
        mal_y_input = [1 for idx in range(len(mal_splits))]

        ### BENIGN
        #benign_all_vec = pickle.load(open('robustness_spec/seed_test_benign/mutate_delete_one/pickles/vectors_all.pickle', 'rb'))
        #benign_splits = pickle.load(open('robustness_spec/seed_test_benign/mutate_delete_one/pickles/splits.pickle', 'rb'))
        #benign_y_input = pickle.load(open('robustness_spec/mutate_delete_one_test_benign/pickles/y_input.pickle', 'rb'))
        #benign_y_input = [0 for idx in range(len(benign_splits))]

        #all_y = mal_y_input + benign_y_input
        all_y_pred = []
        mal_y_pred = []
        benign_y_pred = []
        # predict malicious
        plot_loss, y_pred, pre_softmax, softmax = sess.run([model.plot_loss,model.y_pred, model.pre_softmax, model.y_softmax],
                feed_dict={model.x_input:mal_all_vec,
                    model.y_input:np.zeros(len(mal_all_vec), dtype=np.float32)
                    })

        # get the mal_y_pred if any one is malicious
        j = 0
        for cur_cnt in mal_splits:
            cur_mal = False
            for y in y_pred[j:j+cur_cnt]:
                if y == 1:
                    cur_mal = True
            j += cur_cnt
            if cur_mal is True:
                mal_y_pred.append(1)
            else:
                mal_y_pred.append(0)

        # predict benign
        """
        plot_loss, y_pred, pre_softmax, softmax = sess.run([model.plot_loss,model.y_pred, model.pre_softmax, model.y_softmax],
                feed_dict={model.x_input:benign_all_vec,
                    model.y_input:np.zeros(len(benign_all_vec), dtype=np.float32)
                    })

        # get the mal_y_pred if any one is malicious
        j = 0
        for cur_cnt in benign_splits:
            cur_mal = False
            for y in y_pred[j:j+cur_cnt]:
                if y == 1:
                    cur_mal = True
            j += cur_cnt
            if cur_mal is True:
                benign_y_pred.append(1)
            else:
                benign_y_pred.append(0)

        all_y_pred = mal_y_pred + benign_y_pred
        """

        # include the malicious ones.
        #test_acc, test_fpr = eval(all_y, all_y_pred)
        # only thee malicious ones
        test_acc, test_fpr = eval(mal_y_input, mal_y_pred)
        print 'test accuracy: ', test_acc
        print 'test FPR: ', test_fpr



        """
        x_input = pickle.load(open('robustness_spec/mutate_delete_one_test_benign/pickles/x_input.pickle', 'rb'))
        if type(x_input[0]) == scipy.sparse.csr.csr_matrix:
            x_input = np.array([item.toarray()[0] for item in x_input])


        plot_loss, y_pred, pre_softmax, softmax = sess.run([model.plot_loss,model.y_pred, model.pre_softmax, model.y_softmax],
                feed_dict={model.x_input:x_input,
                    model.y_input:np.zeros(len(x_input), dtype=np.float32)
                    })
        """



if __name__=='__main__':
    args = parse_args()
    main(args)
