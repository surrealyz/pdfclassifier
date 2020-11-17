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

def parse_args():
    parser = argparse.ArgumentParser(description='Regular training and robust training of the pdf malware classification model.')
    parser.add_argument('--train', type=str, help='Training interval data.')
    parser.add_argument('--test', type=str, help='Testing interval data.', required=True)
    parser.add_argument('--test_batches', type=int, help='Number of testing batches', required=True)
    parser.add_argument('--seed_feat', type=str, help='Seed feature value pickle.')
    parser.add_argument('--exploit_spec', type=str, help='Exploit specification file.')
    parser.add_argument('--model_name', type=str, help='Save to this model.', default='test_model_name')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=50)
    # number of batches = epoch * (total_dataset_size / batch_size)
    parser.add_argument('--batches', type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lrdecay", type=float, default=1)
    parser.add_argument('--baseline', action='store_true', default=False)
    parser.add_argument('--verbose', type=int, default=500)
    parser.add_argument('--portion', type=float, default=3)
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


def train(model):
    PATH = '../models/adv_trained/%s.ckpt' % args.model_name
    batch_size = args.batch_size
    batch_num = args.batches
    lr = args.lr

    learning_rate = tf.placeholder(tf.float32)

    loss = model.xent

    optimizer_op = tf.train.AdamOptimizer(learning_rate=\
                            learning_rate).minimize(loss)

    print 'Loading regular training datasets...'
    train_data = '../data/traintest_all_500test/train_data.libsvm'
    x_train, y_train = datasets.load_svmlight_file(train_data,
                                       n_features=3514,
                                       multilabel=False,
                                       zero_based=False,
                                       query_id=False)

    x_train = x_train.toarray()

    print 'Shuffle the training datasets...'
    x_train, y_train = shuffle_data(x_train, y_train)

    print 'Loading regular testing datasets...'
    test_data = '../data/traintest_all_500test/test_data.libsvm'
    x_test, y_test = datasets.load_svmlight_file(test_data,
                                       n_features=3514,
                                       multilabel=False,
                                       zero_based=False,
                                       query_id=False)
    x_test = x_test.toarray()

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
        epoch = 0
        for cur_batch in range(batch_num):
            start_time = time.time()
            x_batch = x_train[j:j+batch_size]
            y_batch = y_train[j:j+batch_size]
            l, acc, fpr, op = sess.run([loss, model.accuracy_op,\
                    model.false_positive_op, optimizer_op],\
                    feed_dict={model.x_input:x_batch,
                        model.y_input:y_batch,
                        learning_rate:lr}
                                )
            # start over.
            j += batch_size
            if j+batch_size > x_train.shape[0]:
                j = 0
                epoch += 1
                # number of batches = epoch * (total_dataset_size / batch_sizie
                if(epoch != 0 and epoch%10==0):
                    lr*=args.lrdecay
                    print "epoch:", epoch, " loss:",l, "train acc:", acc, "train fpr:", fpr, "epoch time:", time.time()-start_time

                if(epoch != 0 and epoch%20==0):
                    test_acc, test_fpr = eval(x_test, y_test, sess, model)
                    print "epoch:", epoch, "eval test acc:", test_acc, "eval test fpr:", test_fpr

        epoch = batch_num * batch_size / x_train.shape[0]
        print "epoch:", epoch, " loss:",l, "train acc:", acc, "epoch time:", time.time()-start_time

        test_acc, test_fpr = eval(x_test, y_test, sess, model)
        print "epoch:", epoch, "eval test acc:", test_acc, "eval test fpr:", test_fpr

        saver.save(sess, save_path=PATH)
        print "Model saved to", PATH

def eval_vra(batch_size, batch_num, x_input, y_input, vectors_all, splits, sess, model):
    start = 0
    end = 0
    print 'Starting prediction to test VRA...'
    y = y_input.tolist()
    y_input_hat = []
    y_input_ipred = []
    for i in range(batch_num):
        end = start + batch_size
        y_pred, ipred, num_correct, ver_acc = sess.run([model.y_pred, model.interval_pred, model.interval_num_correct, model.verified_accuracy],\
                             feed_dict={model.x_input:x_input[start:end],
                                model.y_input:y_input[start:end],
                                model.upper_input:vectors_all[start:end],
                                model.lower_input:x_input[start:end]}

                              )
        # accumulate interval_pred
        y_input_hat += y_pred.tolist()
        y_input_ipred += ipred.tolist()
        start = end

    #compare correct label with predicted label using splits
    j = 0
    total = 0
    acc_correct = 0
    ver_correct = 0
    for cur_cnt in splits:
        if y[j:j+cur_cnt] == y_input_hat[j:j+cur_cnt]:
            acc_correct += 1
        if y[j:j+cur_cnt] == y_input_ipred[j:j+cur_cnt]:
            ver_correct += 1
        total += 1
        j += cur_cnt
        if j+cur_cnt > batch_size*batch_num:
            break
    # last one
    if y[j:] == y_input_hat[j:]:
        acc_correct += 1
    if y[j:] == y_input_ipred[j:]:
        ver_correct += 1
    total += 1
    print total
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
    #PATH = "../models/adv_trained/adv_model.ckpt"
    PATH = '../models/adv_trained/%s.ckpt' % model_name
    batch_size = args.batch_size
    batch_num = args.batches
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

    print 'Shuffle the training datasets...'
    x_train, y_train = shuffle_data(x_train, y_train)

    print 'Loading regular testing datasets...'
    test_data = '../data/traintest_all_500test/test_data.libsvm'
    x_test, y_test = datasets.load_svmlight_file(test_data,
                                       n_features=3514,
                                       multilabel=False,
                                       zero_based=False,
                                       query_id=False)
    x_test = x_test.toarray()


    # load the interval bound datasets
    print 'Loading the training interval datasets...'
    x_input = pickle.load(open(os.path.join(train_interval_path, 'x_input.pickle'), 'rb'))
    y_input = pickle.load(open(os.path.join(train_interval_path, 'y_input.pickle'), 'rb'))
    vectors_all = pickle.load(open(os.path.join(train_interval_path, 'vectors_all.pickle'), "rb"))


    # Load the test data
    print 'Loading the testing interval datasets...'
    x_input_test = pickle.load(open(os.path.join(test_interval_path, 'x_input.pickle'), 'rb'))
    y_input_test = pickle.load(open(os.path.join(test_interval_path, 'y_input.pickle'), 'rb'))
    splits_test = pickle.load(open(os.path.join(test_interval_path, 'splits.pickle'), 'rb'))
    vectors_all_test = pickle.load(open(os.path.join(test_interval_path, 'vectors_all.pickle'), "rb"))

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
        for cur_batch in range(batch_num):
            start_time = time.time()

            # mix train, portion/10 fraction do regular training
            if(cur_batch % 10 < args.portion):
                # regular training
                # This is not the correct fpr
                #print "Running regular training..."
                reg_l, reg_acc, fpr, op = sess.run([regular_loss, model.accuracy_op,\
                    model.false_positive_op, optimizer_op],\
                    feed_dict={model.x_input:x_train[j:j+batch_size],
                        model.y_input:y_train[j:j+batch_size],
                        model.lower_input:x_train[j:j+batch_size],
                        model.upper_input:x_train[j:j+batch_size],
                        learning_rate:lr}
                                )
                j += batch_size
                if j+batch_size > x_train.shape[0]:
                    j = 0
            else:
                # verifiably robust training
                eq, int_l, acc, op = sess.run([model.equation, interval_loss, model.accuracy_op, optimizer_op],\
                        feed_dict={model.x_input:x_input[i:i+batch_size],
                            model.y_input:y_input[i:i+batch_size],
                            model.lower_input:x_input[i:i+batch_size],
                            model.upper_input:vectors_all[i:i+batch_size],
                                            learning_rate:lr}
                                         )
                i += batch_size
                if i+batch_size > x_input.shape[0]:
                    i = 0
                    epoch += 1
                    # display vra
                    eval_vra(batch_size, args.test_batches, x_input_test, y_input_test, vectors_all_test, splits_test, sess, model)
                    acc, fpr = eval(x_test, y_test, sess, model)
                    print "======= test acc:", acc, "test fpr:", fpr

            if cur_batch != 0 and cur_batch % args.verbose ==0:
                lr*=args.lrdecay
                print "batch_num:", cur_batch, "regular loss:", reg_l, "interval loss:",int_l, "regular train acc:", reg_acc , "epoch time:", time.time()-start_time
                acc, fpr = eval(x_test, y_test, sess, model)
                print "*** test acc:", acc, "test fpr:, ", fpr

        print '======= DONE ======='
        eval_vra(batch_size, args.test_batches, x_input_test, y_input_test, vectors_all_test, splits_test, sess, model)
        acc, fpr = eval(x_test, y_test, sess, model)
        print "======= test acc:", acc, "test fpr:", fpr

        saver.save(sess, save_path=PATH)
        print "Model saved to", PATH



def main(args):

    # Initialize the model
    model = Model()

    if(not args.evaluate):
        if(args.baseline):
            train(model)
            return
        else:
            adv_train(model, args.train, args.test, args.model_name)
            return

    if(args.baseline):
        # BASEDLINE FOR EXPERIMENT
        PATH = "../models/adv_trained/baseline_checkpoint.ckpt"
    else:
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

        y_input_test = pickle.load(open(os.path.join(test_interval_path, 'y_input.pickle'), 'rb'))
        splits_test = pickle.load(open(os.path.join(test_interval_path, 'splits.pickle'), 'rb'))
        vectors_all_test = pickle.load(open(os.path.join(test_interval_path, 'vectors_all.pickle'), "rb"))

        seed_dict = pickle.load(open(args.seed_feat, 'rb'))
        exploit_spec = pickle.load(open(args.exploit_spec, 'rb'))
        x_input_test = []
        # the sequence depends on exploit_spec traversal
        idx = 0
        for seed_sha1, exploit_paths in exploit_spec.iteritems():
            if exploit_paths is None:
                continue
            try:
                seed_feature = seed_dict[seed_sha1].toarray()[0]
            except KeyError:
                # this seed_fname can be parsed by pdfrw, but not hidost.
                continue

            for i in range(splits_test[idx]):
                x_input_test.append(seed_feature)
            idx += 1

        x_input_test = np.array(x_input_test)
        print 'Number of intervals for x_input_test:'
        print x_input_test.shape
        print 'Evaluating VRA...'
        # 15752 / 50.0 = 315.04
        eval_vra(args.batch_size, args.test_batches, x_input_test, y_input_test, vectors_all_test, splits_test, sess, model)

if __name__=='__main__':
    args = parse_args()
    main(args)
