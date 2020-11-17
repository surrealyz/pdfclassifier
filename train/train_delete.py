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
import pdfrw
import scipy
import datetime
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Regular training and robust training of the pdf malware classification model.')
    parser.add_argument('--train', type=str, help='Training interval data.')
    parser.add_argument('--test', type=str, help='Testing interval data.', required=True)
    parser.add_argument('--test_batches', type=int, help='Number of testing batches', required=True)
    parser.add_argument('--model_name', type=str, help='Save to this model.')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lrdecay", type=float, default=1)
    parser.add_argument('--verbose', type=int, default=200)
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

def eval_vra_all(batch_size, batch_num, x_input, y_input, vectors_all, splits, sess, model):
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


def shuffle_data(x, y):
    idx = np.arange(0 , len(x))
    np.random.shuffle(idx)
    x_shuffle = np.array([x[i] for i in idx])
    y_shuffle = np.array([y[i] for i in idx])
    return x_shuffle, y_shuffle


def adv_train(model, train_interval_path, test_interval_path, model_name):
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
    print 'Loading the training interval datasets...'
    x_input = pickle.load(open(os.path.join(train_interval_path, 'x_input.pickle'), 'rb'))
    if type(x_input[0]) == scipy.sparse.csr.csr_matrix:
        x_input = np.array([item.toarray()[0] for item in x_input])
    y_input = pickle.load(open(os.path.join(train_interval_path, 'y_input.pickle'), 'rb'))
    vectors_all = pickle.load(open(os.path.join(train_interval_path, 'vectors_all.pickle'), "rb"))

    print x_input.shape

    # Load the test data
    print 'Loading the testing interval datasets...'
    x_input_test = pickle.load(open(os.path.join(test_interval_path, 'x_input.pickle'), 'rb'))
    if type(x_input_test[0]) == scipy.sparse.csr.csr_matrix:
        x_input_test = np.array([item.toarray()[0] for item in x_input_test])
    y_input_test = pickle.load(open(os.path.join(test_interval_path, 'y_input.pickle'), 'rb'))
    splits_test = pickle.load(open(os.path.join(test_interval_path, 'splits.pickle'), 'rb'))
    vectors_all_test = pickle.load(open(os.path.join(test_interval_path, 'vectors_all.pickle'), "rb"))

    print x_input_test.shape

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if(args.resume):
            saver.restore(sess, PATH)
            print "load model from:", PATH
        else:
            print "initial model as:", PATH

        j = 0
        i = 0
        epoch = 0
        for epoch in range(20):
            start_time = time.time()
            # shuffle dataset within every epoch.
            print 'Shuffle the regular training datasets...'
            x_train, y_train = shuffle_data(x_train, y_train)
            # y_input is always ones.
            print 'Shuffle the interval training datasets...'
            x_input, vectors_all = shuffle_data(x_input, vectors_all)


            # regular training index
            j = 0
            # robust training index
            i = 0
            #batch_num = int(len(x_train) / batch_size)
            #print "Running regular training..."
            #for cur_batch in range(batch_num):

            # Can I alternate these batches?
            robust_train_batch = [False for k in range(len(x_train)/batch_size+1)] + [True for k in range(len(x_input)/batch_size+1)]
            b1 = len(x_train)/batch_size+1
            b2 = len(x_input)/batch_size+1
            print 'regular batches:', b1, 'robust batches:', b2
            print 'total batches to run each epoch:', len(robust_train_batch)

            random.shuffle(robust_train_batch)
            for k in range(len(robust_train_batch)):
                if robust_train_batch[k] is False:
                    if j+batch_size > x_train.shape[0]:
                        # last batch within this epoch
                        #print 'batch_size:', batch_size
                        #print 'j:', j

                        remain_size = batch_size-(len(x_train)-j)
                        last_x_train = np.concatenate((x_train[j:], x_train[:remain_size]))
                        last_y_train = np.concatenate((y_train[j:], y_train[:remain_size]))

                        #print 'last regular training batch size:'
                        #print size(last_x_train)
                        #print size(last_y_train)

                        reg_l, reg_acc, fpr, op = sess.run([regular_loss, model.accuracy_op,\
                            model.false_positive_op, optimizer_op],\
                            feed_dict={model.x_input:last_x_train,
                                model.y_input:last_y_train,
                                model.upper_input:x_train[:batch_size],
                                model.lower_input:x_train[:batch_size],
                                learning_rate:lr}
                                        )
                    else:
                        # regular training
                        reg_l, reg_acc, fpr, op = sess.run([regular_loss, model.accuracy_op,\
                            model.false_positive_op, optimizer_op],\
                            feed_dict={model.x_input:x_train[j:j+batch_size],
                                model.y_input:y_train[j:j+batch_size],
                                model.upper_input:x_train[j:j+batch_size],
                                model.lower_input:x_train[j:j+batch_size],
                                learning_rate:lr}
                                        )
                        j += batch_size
                else:
                # robust training
                #i = 0
                #print "Running robust training..."
                #for cur_batch in range(int(len(x_input) / batch_size)):
                    if i+batch_size > x_input.shape[0]:
                        # last batch within this epoch
                        # go to the beginning of x_input
                        remain_size = batch_size-(len(x_input)-i)
                        last_x_input = np.concatenate((x_input[i:], x_input[:remain_size]))
                        last_x_lower = np.concatenate((vectors_all[i:], vectors_all[:remain_size]))
                        last_y_input = np.concatenate((y_input[i:], y_input[:remain_size]))
                        #print 'last robust training batch size:'
                        #print size(last_x_input)
                        #print size(last_x_lower)
                        #print size(last_y_input)

                        eq, int_l, acc, op = sess.run([model.equation, interval_loss, model.accuracy_op, optimizer_op],\
                                feed_dict={model.x_input:last_x_input,
                                    model.y_input:last_y_input,
                                    model.upper_input:last_x_input,
                                    model.lower_input:last_x_lower,
                                                    learning_rate:lr}
                                                 )
                    else:
                        eq, int_l, acc, op = sess.run([model.equation, interval_loss, model.accuracy_op, optimizer_op],\
                                feed_dict={model.x_input:x_input[i:i+batch_size],
                                    model.y_input:y_input[i:i+batch_size],
                                    model.upper_input:x_input[i:i+batch_size],
                                    model.lower_input:vectors_all[i:i+batch_size],
                                                    learning_rate:lr}
                                                 )
                        i += batch_size
                if k != 0 and k % args.verbose ==0:
                    print "batch_num:", k, "regular loss:", reg_l, "interval loss:",int_l, "regular training acc:", reg_acc, "biased interval training acc:", acc
                    test_acc, test_fpr = eval(x_test, y_test, sess, model)
                    print "*** test acc:", test_acc, "test fpr:, ", test_fpr

            #print 'Current learning rate:', lr
            lr*=args.lrdecay
            # FINISHED ONE EPOCH
            print 'Finished epoch %d...' % (epoch+1)
            # display loss values
            print "regular loss:", reg_l, "regular training acc:", reg_acc , "interval loss:", int_l, "biased interval training acc:", acc, "epoch time:", time.time()-start_time
            # display test acc, test fpr, vra
            test_acc, test_fpr = eval(x_test, y_test, sess, model)
            print "======= test acc:", test_acc, "test fpr:", test_fpr
            eval_vra_all(batch_size, args.test_batches, x_input_test, y_input_test, vectors_all_test, splits_test, sess, model)

        #print '======= DONE ======='
        #acc, fpr = eval(x_test, y_test, sess, model)
        #print "======= test acc:", acc, "test fpr:", fpr
        #eval_vra_all(batch_size, args.test_batches, x_input_test, y_input_test, vectors_all_test, splits_test, sess, model)

        saver.save(sess, save_path=PATH)
        print "Model saved to", PATH



def main(args):

    # Initialize the model

    model = Model()

    if(not args.evaluate):
        adv_train(model, args.train, args.test, args.model_name)
        return

    PATH = '../models/adv_trained/%s.ckpt' % args.model_name

    if PATH is None:
        print('No model found')
        sys.exit()

    model.tf_interval1(args.batch_size)
    test_interval_path = args.test

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, PATH)
        print "load model from:", PATH

        # Load the test data
        # some x_input contains items of sparse matrix type
        x_input_test = pickle.load(open(os.path.join(test_interval_path, 'x_input.pickle'), 'rb'))
        if type(x_input_test[0]) == scipy.sparse.csr.csr_matrix:
            x_input_test = np.array([item.toarray()[0] for item in x_input_test])
        y_input_test = pickle.load(open(os.path.join(test_interval_path, 'y_input.pickle'), 'rb'))
        splits_test = pickle.load(open(os.path.join(test_interval_path, 'splits.pickle'), 'rb'))
        vectors_all_test = pickle.load(open(os.path.join(test_interval_path, 'vectors_all.pickle'), "rb"))

        print 'Number of intervals for x_input_test:'
        print x_input_test.shape
        print 'Evaluating VRA...'
        eval_vra_all(args.batch_size, args.test_batches, x_input_test, y_input_test, vectors_all_test, splits_test, sess, model)
        print(datetime.datetime.now())

if __name__=='__main__':
    args = parse_args()
    main(args)
