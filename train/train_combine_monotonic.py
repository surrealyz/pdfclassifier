#! /usr/bin/env python
import os
import time
import argparse
import numpy as np
import pickle
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from model import Model
import datetime
import sys
import random
import scipy

def parse_args():
    parser = argparse.ArgumentParser(description='Regular training and robust training of the pdf malware classification model.')
    parser.add_argument('--train', type=str, help='Training insertion interval data.')
    parser.add_argument('--model_name', type=str, help='Save to this model.', default='test_model_name')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lrdecay", type=float, default=1)
    parser.add_argument('--verbose', type=int, default=300)
    return parser.parse_args()

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def eval(x, y, sess, model):
    y_p = sess.run(model.y_pred,\
                    feed_dict={model.x_input:x,\
                    model.y_input:y
                    })

    try:
        tn, fp, fn, tp = confusion_matrix(y, y_p).ravel()
        acc = (tp+tn)/float(tp+tn+fp+fn)
        fpr = fp/float(fp+tn)
        return acc, fpr
    except ValueError:
        return accuracy_score(y, y_p), None

def acc_fpr(y, y_p):
    try:
        tn, fp, fn, tp = confusion_matrix(y, y_p).ravel()
        print 'tn:', tn, ', fp:', fp, ',fn:', fn, 'tp:', tp
        acc = (tp+tn)/float(tp+tn+fp+fn)
        fpr = fp/float(fp+tn)
        return acc, fpr
    except ValueError:
        return accuracy_score(y, y_p), None


def eval_vra_ins(batch_size, batch_num, x_input, y_input, vectors_all, splits, sess, model):
    start = 0
    end = 0
    print 'Starting prediction to test VRA...'
    y = y_input.tolist()
    y_input_hat = []
    y_input_ipred = []
    # insertion upper bound
    x_upper = vectors_all
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

    last_x_upper = np.concatenate((last_x_upper, np.array([np.zeros(len(x_input[0])) for i in range(remain_size, 50)])))
    y_pred, ipred, num_correct, ver_acc = sess.run([model.y_pred, model.interval_pred, model.interval_num_correct, model.verified_accuracy],\
                         feed_dict={model.x_input:last_x_input,
                            model.y_input:last_y_input,
                            model.upper_input:last_x_upper,
                            model.lower_input:last_x_input}
                          )
    y_input_hat += y_pred.tolist()[:remain_size]
    y_input_ipred += ipred.tolist()[:remain_size]

    print('Total intervals evaluated:', len(y_input_ipred))

    #compare correct label with predicted label using splits
    j = 0
    total = 0
    acc_correct = 0
    ver_correct = 0
    for cur_cnt in splits:
        #for k in range(j, j+cur_cnt):
        if y[j:j+cur_cnt] == y_input_hat[j:j+cur_cnt]:
            acc_correct += 1
        if y[j:j+cur_cnt] == y_input_ipred[j:j+cur_cnt]:
            ver_correct += 1
        j += cur_cnt
        total += 1
    print 'Total splits checked:', total
    final_acc = acc_correct/float(total)
    final_ver_acc = ver_correct/float(total)
    print '======= acc:', final_acc, "ver_acc:", final_ver_acc

def eval_vra_del(batch_size, batch_num, x_input, y_input, vectors_all, splits, sess, model):
    start = 0
    end = 0
    print 'Starting prediction to test VRA...'
    y = y_input.tolist()
    y_input_hat = []
    y_input_ipred = []
    # insertion upper bound
    x_lower = vectors_all
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

    #compare correct label with predicted label using splits
    j = 0
    total = 0
    acc_correct = 0
    ver_correct = 0
    for cur_cnt in splits:
        #for k in range(j, j+cur_cnt):
        if y[j:j+cur_cnt] == y_input_hat[j:j+cur_cnt]:
            acc_correct += 1
        if y[j:j+cur_cnt] == y_input_ipred[j:j+cur_cnt]:
            ver_correct += 1
        j += cur_cnt
        total += 1
    print 'Total splits checked:', total
    final_acc = acc_correct/float(total)
    final_ver_acc = ver_correct/float(total)
    print '======= acc:', final_acc, "ver_acc:", final_ver_acc


def eval_vra_monotonic(batch_size, batch_num, x_input, y_input, sess, model):
    start = 0
    end = 0
    print 'Starting prediction to test VRA...'
    y = y_input.tolist()
    y_input_hat = []
    y_input_ipred = []
    # generate upper_input to be all 1s.
    x_upper = np.array([np.ones(len(x_input[0])) for item in x_input])
    for i in range(batch_num):
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
    #print 'shape', last_x_input.shape
    last_x_input = np.concatenate((last_x_input, np.array([np.zeros(len(x_input[0])) for i in range(remain_size, 50)])))
    last_y_input = np.concatenate((last_y_input, np.array([0 for i in range(remain_size, 50)])))
    #print 'last_x_input shape', last_x_input.shape
    #print 'last_y_input shape', last_y_input.shape

    last_x_upper = np.array([np.ones(len(x_input[0])) for item in last_x_input])
    y_pred, ipred, num_correct, ver_acc = sess.run([model.y_pred, model.interval_pred, model.interval_num_correct, model.verified_accuracy],\
                         feed_dict={model.x_input:last_x_input,
                            model.y_input:last_y_input,
                            model.upper_input:last_x_upper,
                            model.lower_input:last_x_input}
                          )
    y_input_hat += y_pred.tolist()[:remain_size]
    y_input_ipred += ipred.tolist()[:remain_size]

    # acc
    acc, fpr = acc_fpr(y, y_input_hat)
    print('======= acc:', acc)

    # ver_acc
    vra, fpr2 = acc_fpr(y, y_input_ipred)
    print('======= ver_acc:', vra)


def shuffle_data(x, y):
    idx = np.arange(0 , len(x))
    np.random.shuffle(idx)
    x_shuffle = np.array([x[i] for i in idx])
    y_shuffle = np.array([y[i] for i in idx])
    return x_shuffle, y_shuffle

def generate_intervals(feat, spec):
    seed_dict = pickle.load(open(feat, 'rb'))
    exploit_spec = pickle.load(open(spec, 'rb'))
    x_input = []
    for seed_sha1, exploit_paths in exploit_spec.iteritems():
        if exploit_paths is None:
            continue
        try:
            seed_feature = seed_dict[seed_sha1].toarray()[0]
        except KeyError:
            # this seed_fname can be parsed by pdfrw, but not hidost.
            continue
        x_input.append(seed_feature)

    x_input = np.array(x_input)
    x_upper = np.array([np.ones(len(x_input[0])) for item in x_input])
    y_input = np.ones(len(x_input))
    return x_input, x_upper, y_input


def adv_train(model, train_interval_path, model_name):
    PATH = '../models/adv_trained/%s.ckpt' % model_name
    batch_size = args.batch_size
    lr = args.lr

    learning_rate = tf.placeholder(tf.float32)

    #for regular training
    regular_loss = model.xent
    #optimizer_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(regular_loss)

    # for robust training
    model.tf_interval1(batch_size)
    interval_loss = model.interval_xent
    #interval_optimizer_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(interval_loss)

    # try combined loss
    optimizer_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss = regular_loss + interval_loss)

    print 'Loading regular training datasets...'
    train_data = '../data/traintest_all_500test/train_data.libsvm'
    x_train, y_train = datasets.load_svmlight_file(train_data,
                                       n_features=3514,
                                       multilabel=False,
                                       zero_based=False,
                                       query_id=False)
    x_train = x_train.toarray()

    print x_train.shape

    print 'Loading regular testing datasets...'
    test_data = '../data/traintest_all_500test/test_data.libsvm'
    x_test, y_test = datasets.load_svmlight_file(test_data,
                                       n_features=3514,
                                       multilabel=False,
                                       zero_based=False,
                                       query_id=False)
    x_test = x_test.toarray()

    print x_test.shape

    # load the interval bound datasets
    print 'Loading the insertion training interval datasets...'
    pickle_dir = args.train
    ins_x_input = pickle.load(open(os.path.join(pickle_dir, 'x_input.pickle'), 'rb'))
    if type(ins_x_input[0]) == scipy.sparse.csr.csr_matrix:
        ins_x_input = np.array([item.toarray()[0] for item in ins_x_input])
    ins_y_input = pickle.load(open(os.path.join(pickle_dir, 'y_input.pickle'), 'rb'))
    ins_vectors_all = pickle.load(open(os.path.join(pickle_dir, 'vectors_all.pickle'), "rb"))

    print ins_x_input.shape

    # Load the test data
    print 'Loading the insertion testing interval datasets...'
    test_pickle_dir = 'robustness_spec/seed_test_malicious/mutate_insert_rootany/pickles/'
    ins_x_input_test = pickle.load(open(os.path.join(test_pickle_dir, 'x_input.pickle'), 'rb'))
    if type(ins_x_input_test[0]) == scipy.sparse.csr.csr_matrix:
        ins_x_input_test = np.array([item.toarray()[0] for item in ins_x_input_test])
    ins_y_input_test = pickle.load(open(os.path.join(test_pickle_dir, 'y_input.pickle'), 'rb'))
    ins_splits_test = pickle.load(open(os.path.join(test_pickle_dir, 'splits.pickle'), 'rb'))
    ins_vectors_all_test = pickle.load(open(os.path.join(test_pickle_dir, 'vectors_all.pickle'), "rb"))

    print ins_x_input_test.shape

    print 'Loading the deletion training interval datasets...'
    pickle_dir = 'robustness_spec/seed_train_malicious/mutate_delete_one/pickles/'
    del_x_input = pickle.load(open(os.path.join(pickle_dir, 'x_input.pickle'), 'rb'))
    if type(del_x_input[0]) == scipy.sparse.csr.csr_matrix:
        del_x_input = np.array([item.toarray()[0] for item in del_x_input])
    del_y_input = pickle.load(open(os.path.join(pickle_dir, 'y_input.pickle'), 'rb'))
    del_vectors_all = pickle.load(open(os.path.join(pickle_dir, 'vectors_all.pickle'), "rb"))

    print 'Loading the deletion testing interval datasets...'
    test_pickle_dir = 'robustness_spec/seed_test_malicious/mutate_delete_one/pickles/'
    del_x_input_test = pickle.load(open(os.path.join(test_pickle_dir, 'x_input.pickle'), 'rb'))
    if type(del_x_input_test[0]) == scipy.sparse.csr.csr_matrix:
        del_x_input_test = np.array([item.toarray()[0] for item in del_x_input_test])
    del_y_input_test = pickle.load(open(os.path.join(test_pickle_dir, 'y_input.pickle'), 'rb'))
    del_splits_test = pickle.load(open(os.path.join(test_pickle_dir, 'splits.pickle'), 'rb'))
    del_vectors_all_test = pickle.load(open(os.path.join(test_pickle_dir, 'vectors_all.pickle'), "rb"))

    # generate training and testing intervals
    print 'Generating the monotonic interval datasets'
    train_feat = 'robustness_spec/seed_train_malicious/feat_dict.pickle'
    train_spec = '../data/traintest_all_500test/exploit_spec/train_malicious.pickle'
    train_batches = 137
    test_feat = 'robustness_spec/seed_test_malicious/feat_dict.pickle'
    test_spec = '../data/traintest_all_500test/exploit_spec/test_malicious.pickle'
    test_batches = 68
    print 'Generating the training interval datasets...'
    m_x_input, m_x_upper, m_y_input = generate_intervals(train_feat, train_spec)
    print 'Generating the testing interval datasets...'
    m_x_input_test, m_x_upper_test, m_y_input_test = generate_intervals(test_feat, test_spec)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if(args.resume):
            saver.restore(sess, PATH)
            print "load model from:", PATH
        else:
            print "initial model as:", PATH

        for epoch in range(10):
            start_time = time.time()
            # shuffle dataset within every epoch.
            print 'Shuffle the regular training datasets...'
            x_train, y_train = shuffle_data(x_train, y_train)
            # y_input is always ones.
            print 'Shuffle the insertion intervals...'
            ins_x_input, ins_vectors_all = shuffle_data(ins_x_input, ins_vectors_all)
            print 'Shuffle the deletion intervals...'
            del_x_input, del_vectors_all = shuffle_data(del_x_input, del_vectors_all)

            # regular training index
            j = 0
            # robust deletion training index
            i = 0
            # robust insertion training index
            k = 0
            # robust monotonic training index
            m = 0
            b1 = len(x_train)/batch_size+1
            b2 = len(ins_x_input)/batch_size+1
            b3 = len(del_x_input)/batch_size+1
            b4 = len(m_x_input)/batch_size+1
            robust_train_batch = [0 for s in range(b1)] + [1 for s in range(b2)] + [2 for s in range(b3)] + [3 for t in range(b4)]
            print 'regular batches:', b1, 'insertion batches:', b2, 'deletion batches:', b3, 'monotonic batches:', b4
            print 'total batches to run each epoch:', len(robust_train_batch)

            random.shuffle(robust_train_batch)
            for s in range(len(robust_train_batch)):
                # regular training
                if robust_train_batch[s] == 0:
                    if j+batch_size > x_train.shape[0]:
                        remain_size = batch_size-(len(x_train)-j)
                        last_x_train = np.concatenate((x_train[j:], x_train[:remain_size]))
                        last_y_train = np.concatenate((y_train[j:], y_train[:remain_size]))
                        reg_l, reg_acc, fpr, op = sess.run([regular_loss, model.accuracy_op,\
                            model.false_positive_op, optimizer_op],\
                            feed_dict={model.x_input:last_x_train,
                                model.y_input:last_y_train,
                                model.upper_input:x_train[:batch_size],
                                model.lower_input:x_train[:batch_size],
                                learning_rate:lr}
                                        )
                    else:
                        reg_l, reg_acc, fpr, op = sess.run([regular_loss, model.accuracy_op,\
                            model.false_positive_op, optimizer_op],\
                            feed_dict={model.x_input:x_train[j:j+batch_size],
                                model.y_input:y_train[j:j+batch_size],
                                model.upper_input:x_train[j:j+batch_size],
                                model.lower_input:x_train[j:j+batch_size],
                                learning_rate:lr}
                                        )
                        j += batch_size
                # insertion robust training
                elif robust_train_batch[s] == 1:
                    if i+batch_size > ins_x_input.shape[0]:
                        # last batch within this epoch
                        # go to the beginning of x_input
                        remain_size = batch_size-(len(ins_x_input)-i)
                        last_x_input = np.concatenate((ins_x_input[i:], ins_x_input[:remain_size]))
                        last_x_upper = np.concatenate((ins_vectors_all[i:], ins_vectors_all[:remain_size]))
                        last_y_input = np.concatenate((ins_y_input[i:], ins_y_input[:remain_size]))
                        eq, int_l, acc, op = sess.run([model.equation, interval_loss, model.accuracy_op, optimizer_op],\
                                feed_dict={model.x_input:last_x_input,
                                    model.y_input:last_y_input,
                                    model.upper_input:last_x_upper,
                                    model.lower_input:last_x_input,
                                                    learning_rate:lr}
                                                 )
                    else:
                        eq, int_l, acc, op = sess.run([model.equation, interval_loss, model.accuracy_op, optimizer_op],\
                                feed_dict={model.x_input:ins_x_input[i:i+batch_size],
                                    model.y_input:ins_y_input[i:i+batch_size],
                                    model.upper_input:ins_vectors_all[i:i+batch_size],
                                    model.lower_input:ins_x_input[i:i+batch_size],
                                                    learning_rate:lr}
                                                 )
                        i += batch_size
                # deletion robust training
                elif robust_train_batch[s] == 2:
                    if k+batch_size > del_x_input.shape[0]:
                        # last batch within this epoch
                        # go to the beginning of x_input
                        remain_size = batch_size-(len(del_x_input)-k)
                        last_x_input = np.concatenate((del_x_input[k:], del_x_input[:remain_size]))
                        last_x_lower = np.concatenate((del_vectors_all[k:], del_vectors_all[:remain_size]))
                        last_y_input = np.concatenate((del_y_input[k:], del_y_input[:remain_size]))
                        eq, int_l, acc, op = sess.run([model.equation, interval_loss, model.accuracy_op, optimizer_op],\
                                feed_dict={model.x_input:last_x_input,
                                    model.y_input:last_y_input,
                                    model.upper_input:last_x_input,
                                    model.lower_input:last_x_lower,
                                                    learning_rate:lr}
                                                 )
                    else:
                        eq, int_l, acc, op = sess.run([model.equation, interval_loss, model.accuracy_op, optimizer_op],\
                                feed_dict={model.x_input:del_x_input[k:k+batch_size],
                                    model.y_input:del_y_input[k:k+batch_size],
                                    model.upper_input:del_x_input[k:k+batch_size],
                                    model.lower_input:del_vectors_all[k:k+batch_size],
                                                    learning_rate:lr}
                                                 )
                        k += batch_size
                else:
                    if m+batch_size > m_x_input.shape[0]:
                        # last batch within this epoch
                        # go to the beginning of x_input
                        remain_size = batch_size-(len(m_x_input)-m)
                        last_x_input = np.concatenate((m_x_input[m:], m_x_input[:remain_size]))
                        last_x_upper = np.concatenate((m_x_upper[m:], m_x_upper[:remain_size]))
                        last_y_input = np.concatenate((m_y_input[m:], m_y_input[:remain_size]))
                        eq, int_l, acc, op = sess.run([model.equation, interval_loss, model.accuracy_op, optimizer_op],\
                                feed_dict={model.x_input:last_x_input,
                                    model.y_input:last_y_input,
                                    model.upper_input:last_x_upper,
                                    model.lower_input:last_x_input,
                                                    learning_rate:lr}
                                                 )
                    else:
                        eq, int_l, acc, op = sess.run([model.equation, interval_loss, model.accuracy_op, optimizer_op],\
                                feed_dict={model.x_input:m_x_input[m:m+batch_size],
                                    model.y_input:m_y_input[m:m+batch_size],
                                    model.upper_input:m_x_upper[m:m+batch_size],
                                    model.lower_input:m_x_input[m:m+batch_size],
                                                    learning_rate:lr}
                                                 )
                        m += batch_size

                if s != 0 and s % args.verbose ==0:
                    print "batch_num:", s, "regular loss:", reg_l, "interval loss:",int_l, "regular training acc:", reg_acc, "biased interval training acc:", acc
                    test_acc, test_fpr = eval(x_test, y_test, sess, model)
                    print "*** test acc:", test_acc, "test fpr:, ", test_fpr

            lr*=args.lrdecay
            # FINISHED ONE EPOCH
            print 'Finished epoch %d...' % (epoch+1)
            test_acc, test_fpr = eval(x_test, y_test, sess, model)
            print "======= test acc:", test_acc, "test fpr:", test_fpr
            # display vra
            print 'INSERTION VRA'
            eval_vra_ins(batch_size, 2869, ins_x_input_test, ins_y_input_test, ins_vectors_all_test, ins_splits_test, sess, model)
            print 'DELETION VRA'
            eval_vra_del(batch_size, 313, del_x_input_test, del_y_input_test, del_vectors_all_test, del_splits_test, sess, model)
            print 'MONOTONIC VRA'
            eval_vra_monotonic(batch_size, test_batches, m_x_input_test, m_y_input_test, sess, model)

            if args.resume:
                cur_path = '../models/adv_trained/%s_e%s.ckpt' % (model_name, epoch+11)
            else:
                cur_path = '../models/adv_trained/%s_e%s.ckpt' % (model_name, epoch+1)
            print '======= SAVING MODELS TO: %s' % cur_path
            saver.save(sess, save_path=cur_path)

        """
        print '======= DONE ======='
        print 'INSERTION VRA'
        eval_vra_ins(batch_size, 2869, ins_x_input_test, ins_y_input_test, ins_vectors_all_test, ins_splits_test, sess, model)
        # display vra
        print 'DELETION VRA'
        eval_vra_del(batch_size, 313, del_x_input_test, del_y_input_test, del_vectors_all_test, del_splits_test, sess, model)
        acc, fpr = eval(x_test, y_test, sess, model)
        print "======= test acc:", acc, "test fpr:", fpr
        """

        saver.save(sess, save_path=PATH)
        print "Model saved to", PATH



def main(args):

    # Initialize the model

    model = Model()

    print(datetime.datetime.now())
    adv_train(model, args.train, args.model_name)
    print(datetime.datetime.now())
    return


if __name__=='__main__':
    args = parse_args()
    main(args)
