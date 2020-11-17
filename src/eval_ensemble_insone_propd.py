#! /usr/bin/env python
import os
import sys
import argparse
import pygtrie
import pickle
import numpy as np
from model import Model
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the robustness of insone ensemble model.Every time, deletion is evaluated. Test property D on the fly.')
    parser.add_argument('--model_name', type=str, help='Load a previous trained model.', required=True)
    return parser.parse_args()

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



def insert_40(sess, model, seed_fname, seed_sha1, seed_feature, feat_trie):
    global vidx
    global hidost_path
    hidost_pathfile = os.path.join(hidost_path, seed_sha1)

    try:
        fin = open(hidost_pathfile, 'r')
    except IOError:
        fin = open(hidost_pathfile+'.pdf', 'r')

    t = pygtrie.StringTrie(separator=os.path.sep)
    #lno = 1
    header = True
    for line in fin:
        if header is True:
            header = False
            continue
        path, cnt = line.rstrip().split(' ')
        path = '%s' % path.replace('\x00', '/').rstrip('/')

    sha1_pred = {}

    ops = []
    allkeys = list(feat_trie._root.children.keys())
    #print allkeys
    total = len(allkeys)
    vec = []
    for s in range(total-1):
        for t in range(s+1, total):
            #print total, s, t
            newvec = np.copy(seed_feature)
            # insert everything except allkeys[s], then delete allkeys[t].
            key1, key2 = allkeys[s], allkeys[t]
            # insert all but not the current key1.
            min_idx, max_idx = feat_trie[key1]
            # change seed feature
            newvec = np.copy(seed_feature)
            # things before min_idx
            for i in range(0, min_idx-1):
                newvec[i] = 1
            # things after max_idx
            for i in range(max_idx, len(newvec)):
                newvec[i] = 1
            #print 'insert:', key, min_idx, max_idx
            # delete key2
            min_idx, max_idx = feat_trie[key2]
            for i in range(min_idx-1, max_idx):
                newvec[i] = 0
            vec.append(newvec)
            ops.append((seed_sha1, key1, key2))
            if sha1_pred.get(seed_sha1, None) is None:
                sha1_pred[seed_sha1] = {}
            # there are repeats but that's fine. so I don't need to keep track of the last one.
            sha1_pred[seed_sha1][key1] = 0
            sha1_pred[seed_sha1][key2] = 0


    # vec is the upper bound
    # make lower bound
    x_input = np.repeat([seed_feature], len(vec), axis=0)
    y_input = np.ones(len(vec),  dtype=np.float32)
    y_pred = classify_intervals_insert(sess, model, x_input, y_input, vec)
    for vidx in range(len(y_pred)):
        pred = y_pred[vidx]
        sha1, sub1, sub2 = ops[vidx]
        if pred == 1:
            sha1_pred[sha1][sub1] = 1
            sha1_pred[sha1][sub2] = 1

    incorrect = 0
    for sha1, all_sub in sha1_pred.iteritems():
        for key, pred in all_sub.iteritems():
            if pred == 0:
                return False

    return True

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

        seed_dir = '../data/traintest_all_500test/test_malicious'

        # load the exploit spec.
        exploit_spec = pickle.load(open('../data/traintest_all_500test/exploit_spec/test_malicious.pickle', 'rb'))
        seed_feat = 'robustness_spec/seed_test_malicious/feat_dict.pickle'
        seed_dict = pickle.load(open(seed_feat, 'rb'))
        feat_trie = pickle.load(open('robustness_spec/feature_spec/pathtrie_filled.pickle', 'rb'))

        global hidost_path
        hidost_path = '../data/extracted_structural_paths/test_malicious'

        print 'Property D'
        global vidx
        vidx = 0
        firstone = True
        splits = []
        cnt = 0
        total = 0
        correct = 0
        # go through all files with exploit spec
        for seed_sha1, exploit_paths in exploit_spec.iteritems():
            if exploit_paths is None:
                continue
            try:
                seed_feature = seed_dict[seed_sha1].toarray()[0]
            except KeyError:
                # this seed_fname can be parsed by pdfrw, but not hidost.
                continue
            # generate intervals for sha1 pdf
            seed_fname = os.path.join(seed_dir, '%s.pdf' % seed_sha1)
            #print seed_fname
            safe = insert_40(sess, model, seed_fname, seed_sha1, seed_feature, feat_trie)
            print safe
            if safe:
                correct += 1
            total += 1
            # print progress
            cnt += 1
            if cnt % 300 == 0:
                print cnt
        print 'total:', total
        print 'correct:', correct



if __name__=='__main__':
    args = parse_args()
    main(args)
