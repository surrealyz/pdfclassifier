#! /usr/bin/env python
import os
import time
import argparse
import numpy as np
import pickle
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow.compat.v1 as tf
from model import Model
import pdb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_args():
    parser = argparse.ArgumentParser(description='Regular training and robust training of the pdf malware classification model.')
    parser.add_argument('--method', type=str, default="A", help="d1, i1, d2, i41, mono")
    parser.add_argument('--model', type=str, default="baseline", help="baseline")
    parser.add_argument('--iters', type=int, default="4000", help="Number of iterations for gradient attacks")
    parser.add_argument('--gpu', type=str, default="0")
    return parser.parse_args()

def eval(x, y, sess, model):
    y_p = sess.run(model.y_pred,\
                    feed_dict={model.x_input:x,\
                    model.y_input:y
                    })

    try:
        tn, fp, fn, tp = confusion_matrix(y, y_p).ravel()
        #print tn, fp, fn, tp
        acc = (tp+tn)/float(tp+tn+fp+fn)
        if((tn+fp)!=0):
            fpr = fp/float(fp+tn)
            return acc, fpr
        else:
            return acc
    except ValueError:
        return accuracy_score(y, y_p), None


def find_model_path(args):
    model_path = "../models/adv_trained/"
    if args.model == "baseline":
        PATH = model_path + "baseline_checkpoint.ckpt"
    elif args.model == "TA":
        PATH = model_path + "baseline_adv_delete_one.ckpt"
    elif args.model == "TB":
        PATH = model_path + "baseline_adv_insert_one.ckpt"
    elif args.model == "TC":
        PATH = model_path + "baseline_adv_delete_two.ckpt"
    elif args.model == "TD":
        PATH = model_path + "baseline_adv_insert_rootallbutone.ckpt"
    elif args.model == "ATAB":
        PATH = model_path + "baseline_adv_combine_two.ckpt"
    elif args.model == "EAB":
        PATH = model_path + "adv_del_twocls.ckpt"
    elif args.model == "ED":
        PATH = model_path + "adv_keep_twocls.ckpt"
    elif args.model == "RA":
        PATH = model_path + "robust_delete_one.ckpt"
    elif args.model == "RB":
        PATH = model_path + "robust_insert_one.ckpt"
    elif args.model == "RC":
        PATH = model_path + "robust_delete_two.ckpt"
    elif args.model == "RD":
        PATH = model_path + "robust_insert_allbutone.ckpt"
    elif args.model == "RAB":
        PATH = model_path + "robust_combine_two_v2_e18.ckpt"
    elif args.model == "RABE":
        PATH = model_path + "robust_combine_three_e17.ckpt"
    elif args.model == "mono":
        PATH = model_path + "robust_monotonic.ckpt"
    else:
        print("no such model!")
        exit()
    return PATH


def cal_grad(sess, model, benign_gradient, attack_loss, x, y):
    g, l, acc = sess.run([benign_gradient, model.y_pred, model.accuracy], feed_dict={model.x_input:x, model.y_input:y})
    return g, l, acc


def find_index(sess, model, benign_gradient, attack_loss, inds_tensor, index_tensor, x, y, inds):
    g, l, acc = sess.run([benign_gradient, model.pre_softmax, model.accuracy], feed_dict={model.x_input:x, model.y_input:y, inds_tensor:inds})
    return g, l, acc


def unrestricted_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie):

    adv_acc = 0
    is_adv = np.zeros(x.shape[0]).astype(bool)
    new_is_adv = np.zeros(x.shape[0]).astype(bool)
    adv_accs = []

    adv_acc = 1.-float(np.sum(is_adv))/float(x.shape[0])
    stuck_acc = -1

    inds = list(range(x.shape[1]))
    x_adv = np.copy(x)
    y_adv = np.copy(y)
    steps = 0
    distances = np.ones(x.shape[0])*(-1.)
    samples = np.copy(x)

    #while np.sum(x_adv[:, inds]==0.):
    for steps in range(200000):
        #start_time = time.time()
        x_adv = x_adv[new_is_adv==0]
        y_adv = y_adv[new_is_adv==0]

        g_adv, l_adv, acc_adv = cal_grad(sess, model, benign_gradient, attack_loss, x_adv, y_adv)
        new_is_adv = np.not_equal(l_adv, y_adv)

        distance = np.sum(np.absolute(x_adv-x[is_adv==0]), axis=1)
        #print(is_adv.shape, distances.shape, distance.shape)
        distances[is_adv==0] = distance
        samples[is_adv==0] = x_adv

        is_adv[is_adv==0] = np.logical_or(is_adv[is_adv==0], new_is_adv)
        adv_acc = 1.-float(np.sum(is_adv))/float(x.shape[0])

        print("steps:", steps, "adv_acc:", adv_acc, "distance:", np.mean(distances))
        adv_accs.append([steps, adv_acc])

        if adv_acc == 0.:
            print("acc break")
            break

        for i in range(g_adv.shape[0]):

            zero_inds = list(np.nonzero(1.-x_adv[i])[0])
            one_inds = list(np.nonzero(x_adv[i])[0])
            r = np.random.rand()
            if (r>0.5) and one_inds:
                zero_inds = []

            if zero_inds and one_inds:
                # nature greedy

                index_zero = np.argmax(g_adv[i, zero_inds])
                index_one = np.argmax(-g_adv[i, one_inds])

                if (g_adv[i, zero_inds][index_zero]) >= (-g_adv[i, one_inds][index_one]):
                    assert x_adv[i][zero_inds[index_zero]] == 0., "insert wrong"
                    x_adv[i][zero_inds[index_zero]] = 1
                else:
                    assert x_adv[i][one_inds[index_one]] == 1., "del wrong"
                    x_adv[i][one_inds[index_one]] = 0


            elif zero_inds:
                index_zero = np.argmax(g_adv[i, zero_inds])
                assert x_adv[i][zero_inds[index_zero]] == 0., "insert wrong"
                x_adv[i][zero_inds[index_zero]] = 1

            elif one_inds:
                index_one = np.argmax(-g_adv[i, one_inds])
                assert x_adv[i][one_inds[index_one]] == 1., "del wrong"
                x_adv[i][one_inds[index_one]] = 0
            else:
                break

        #print(time.time()-start_time)
    g_adv, l_adv, sample_acc = cal_grad(sess, model, benign_gradient, attack_loss, samples, y)
    print("samples acc", sample_acc)
    adv_acc = 1.-float(np.sum(is_adv))/float(x.shape[0])
    #assert sample_acc == adv_acc, "wrong adversarial samples "+str(sample_acc)+"/"+str(adv_acc)
    return adv_accs, distances, samples, is_adv


def unrestricted_delete_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie):
    x_adv = np.copy(x)
    for t in range(args.iters):
        g, l, acc = cal_grad(sess, model, benign_gradient, attack_loss, x_adv, y)
        index = np.argmax(x_adv*(-g), axis=-1)
        for i in range(index.shape[0]):
            x_adv[i][index[i]] = 0
        adv_acc = eval(x_adv, y, sess, model)
        print ("iters:", t, "adv_acc", adv_acc)
    return adv_acc


def unrestricted_insert_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie):

    adv_acc = 0
    is_adv = np.zeros(x.shape[0]).astype(bool)
    new_is_adv = np.zeros(x.shape[0]).astype(bool)
    adv_accs = []

    adv_acc = 1.-float(np.sum(is_adv))/float(x.shape[0])
    stuck_acc = -1

    inds = list(range(x.shape[1]))
    x_adv = np.copy(x)
    y_adv = np.copy(y)
    steps = 0
    distances = np.ones(x.shape[0])*(-1.)
    samples = np.copy(x)

    #while np.sum(x_adv[:, inds]==0.):
    for steps in range(x.shape[1]):
        #start_time = time.time()
        x_adv = x_adv[new_is_adv==0]
        y_adv = y_adv[new_is_adv==0]

        if np.sum(x_adv[:, inds]==0.)==0.:
            break

        g_adv, l_adv, acc_adv = cal_grad(sess, model, benign_gradient, attack_loss, x_adv, y_adv)
        new_is_adv = np.not_equal(l_adv, y_adv)

        distance = np.sum(np.absolute(x_adv-x[is_adv==0]), axis=1)
        distances[is_adv==0] = distance
        samples[is_adv==0] = x_adv

        is_adv[is_adv==0] = np.logical_or(is_adv[is_adv==0], new_is_adv)
        adv_acc = 1.-float(np.sum(is_adv))/float(x.shape[0])

        print("steps:", steps, "adv_acc:", adv_acc, "distance:", np.mean(distances))
        adv_accs.append([steps, adv_acc])

        if adv_acc == 0.:
            print("acc break")
            break

        for i in range(g_adv.shape[0]):
            zero_inds = list(np.nonzero(1.-x_adv[i])[0])
            if not zero_inds:
                break

            index = np.argmax(g_adv[i, zero_inds])

            assert x_adv[i][zero_inds[index]] == 0., "insert wrong"
            x_adv[i][zero_inds[index]] = 1
        #print(time.time()-start_time)

    g_adv, l_adv, sample_acc = cal_grad(sess, model, benign_gradient, attack_loss, samples, y)
    print("samples acc", sample_acc)
    adv_acc = 1.-float(np.sum(is_adv))/float(x.shape[0])
    #assert sample_acc == adv_acc, "wrong adversarial samples "+str(sample_acc)+"/"+str(adv_acc)
    return adv_accs, distances, samples, is_adv


def delete1_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie):
    adv_acc = 0
    break_samples = []
    for i in range(x.shape[0]):
        xi_adv = np.copy(x[i:i+1])
        yi = y[i:i+1]
        is_adv = False
        #print("sample", i)
        for key, value in feat_trie._root.children.iteritems():
            xi_adv = np.copy(x[i:i+1])
            min_idx, max_idx = feat_trie[key]
            #print(i, key, min_idx, max_idx)
            inds = []
            for ind in range(min_idx-1, max_idx):
                if xi_adv[0,ind]==1.:
                    inds.append(ind)
            #if inds: print(key, inds)
            #for t in range(args.iters):
            while inds:
                gi, li, acci = cal_grad(sess, model, benign_gradient, attack_loss, xi_adv, yi)
                if acci == 0.:
                    is_adv = True
                    break
                #x_adv[:, inds]*(-g[:, inds])
                index = np.argmax(-gi[0][inds])
                xi_adv[0,inds[index]] = 0.
                inds.pop(index)

            if is_adv:
                #print("break sample", i, "distance:", np.sum(x[i:i+1] - xi_adv))
                adv_acc += 1
                break_samples.append(i)
                break

    adv_acc = 1.-float(adv_acc)/float(x.shape[0])
    #print("adv_acc: %.2f" % (adv_acc*100.))
    #print(break_samples)

    return adv_acc



def delete2_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie):
    adv_acc = 0
    keys = []
    break_samples = []

    for key, value in feat_trie._root.children.iteritems():
        keys.append(key)

    for i in range(x.shape[0]):
        xi_adv = np.copy(x[i:i+1])
        yi = y[i:i+1]
        is_adv = False
        #print("sample", i)
        for k1 in range(len(keys)-1):
            key1 = keys[k1]
            for k2 in range(k1, len(keys)):
                xi_adv = np.copy(x[i:i+1])
                key2 = keys[k2]
                min_idx1, max_idx1 = feat_trie[key1]
                min_idx2, max_idx2 = feat_trie[key2]
                inds = []
                for ind in range(min_idx1-1, max_idx1):
                    if xi_adv[0,ind]==1.:
                        #print(ind, xi_adv[0,ind])
                        inds.append(ind)

                for ind in range(min_idx2-1, max_idx2):
                    if (xi_adv[0,ind]==1.) and (ind not in inds):
                        #print(ind, xi_adv[0,ind])
                        inds.append(ind)

                while inds:
                    gi, li, acci = cal_grad(sess, model, benign_gradient, attack_loss, xi_adv, yi)
                    if acci == 0.:
                        is_adv = True
                        break
                    index = np.argmax(-gi[0][inds])
                    xi_adv[0,inds[index]] = 0.

                    #print(inds[index], gi[0][inds])
                    #exit()
                    inds.pop(index)

                if is_adv:
                    break

            if is_adv:
                adv_acc += 1
                #print("break sample", i, "distance:", np.sum(x[i:i+1] - xi_adv))
                break_samples.append(i)
                break

    adv_acc = 1.-float(adv_acc)/float(x.shape[0])
    #print(break_samples)

    return adv_acc


def insert1_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie):
    adv_acc = 0
    for i in range(x.shape[0]):
        xi_adv = np.copy(x[i:i+1])
        yi = y[i:i+1]
        is_adv = False
        #print("sample", i)
        for key, value in feat_trie._root.children.iteritems():
            xi_adv = np.copy(x[i:i+1])
            min_idx, max_idx = feat_trie[key]
            #print(i, key, min_idx, max_idx)
            inds = []
            for ind in range(min_idx-1, max_idx):
                if xi_adv[0,ind]==0.:
                    inds.append(ind)

            #for t in range(args.iters):
            while inds:
                gi, li, acci = cal_grad(sess, model, benign_gradient, attack_loss, xi_adv, yi)
                #print("acci", acci)
                if acci == 0.:
                    #print("evade", i)
                    is_adv = True
                    break
                index = np.argmax(gi[0][inds])
                xi_adv[0,inds[index]] = 1.
                inds.pop(index)

            if is_adv:
                adv_acc += 1
                #print("break sample", i, "distance:", np.sum(xi_adv - x[i:i+1]))
                break
    adv_acc = 1.-float(adv_acc)/float(x.shape[0])
    #print("adv_acc: %.2f" % (adv_acc*100.))

    return adv_acc


def insert41_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie):
    adv_acc = 0
    is_adv = np.zeros(x.shape[0]).astype(bool)
    new_is_adv = np.zeros(x.shape[0]).astype(bool)
    keys = []
    values = []
    for key, value in feat_trie._root.children.iteritems():
        min_idx, max_idx = feat_trie[key]
        keys.append(key)
        values.append(max_idx-min_idx)

    keys = np.array(keys)[np.argsort(np.array(values))]


    for key in keys:
        print(key)
        adv_acc = 1.-float(np.sum(is_adv))/float(x.shape[0])
        if adv_acc==0.: break
        stuck_acc = -1
        min_idx, max_idx = feat_trie[key]
        inds = list(range(0, min_idx-1))+list(range(max_idx, x.shape[1]))
        x_adv = np.copy(x[is_adv==0])
        y_adv = np.copy(y[is_adv==0])
        steps = 0
        total_zero = np.sum(x_adv[:,inds]==0.)

        while total_zero>0.:
            total_zero = 0
            total_cut = 0
            #start_time = time.time()
            #if steps >= 100:
                #break
            x_adv = np.copy(x_adv[new_is_adv==0])
            y_adv = np.copy(y_adv[new_is_adv==0])

            min_idx, max_idx = feat_trie[key]

            g_adv, l_adv, acc_adv = cal_grad(sess, model, benign_gradient, attack_loss, x_adv, y_adv)
            new_is_adv = np.not_equal(l_adv, y_adv)
            is_adv[is_adv==0] = np.logical_or(is_adv[is_adv==0], new_is_adv)
            adv_acc = 1.-float(np.sum(is_adv))/float(x.shape[0])
            print("steps:", steps, "adv_acc:", adv_acc)

            if adv_acc == stuck_acc:
                steps += 1
            else:
                stuck_acc = adv_acc
                steps = 0

            if adv_acc == 0.: break

            for i in range(g_adv.shape[0]):
                zero_inds = []
                for ind in inds:
                    if x_adv[i, ind] == 0.:
                        zero_inds.append(ind)
                if not zero_inds:
                    break
                total_zero += len(zero_inds)

                index = np.argmax(g_adv[i, zero_inds])

                assert x_adv[i][zero_inds[index]] == 0., "insert wrong"
                x_adv[i][zero_inds[index]] = 1
                total_cut += 1
            #print(time.time()-start_time)

            #print(total_zero, total_cut, total_zero-total_cut, g_adv.shape[0])


    adv_acc = 1.-float(np.sum(is_adv))/float(x.shape[0])

    return adv_acc



def gradient_attack(sess, model, benign_gradient, attack_loss, x, y):
    g, l = sess.run([benign_gradient, model.pre_softmax], feed_dict={model.x_input:x, model.y_input:y})

    index = np.argmax((1.0-x)*g, axis=-1)

    for i in range(index.shape[0]):
        x[i][index[i]] = 1.0

    #g, l = sess.run([benign_gradient, model.pre_softmax], feed_dict={model.x_input:x, model.y_input:y})
    #print l[:10]

    return x


def attack(args, saver, sess, model, benign_gradient, attack_loss, x, y, feat_trie):

    PATH = find_model_path(args)
    saver.restore(sess, PATH)
    print ("load model from:", PATH)

    acc = eval(x, y, sess, model)
    print( "clean acc", acc)

    import time
    start_time = time.time()

    if args.method == "un":
        print("unstricted attack")
        adv_accs, distances, samples, is_adv = unrestricted_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie)
        return adv_accs, distances, samples, is_adv
    elif args.method == "uni":
        print("unrestricted insert attack")
        adv_accs, distances, samples, is_adv = unrestricted_insert_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie)
        return adv_accs, distances, samples, is_adv
    elif args.method == "und":
        print("unstricted attack")
        adv_acc = unrestricted_delete_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie)
    elif args.method == "A":
        print("delete1 attack")
        adv_acc = delete1_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie)
    elif args.method == "B":
        print("insert1 attack")
        adv_acc = insert1_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie)
    elif args.method == "C":
        print("delete2 attack")
        adv_acc = delete2_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie)
    elif args.method == "D":
        print("insert41 attack")
        adv_acc = insert41_gradient_attack(sess, model, benign_gradient, attack_loss, x, y, feat_trie)
    else:
        print("no such attack method!")
        exit()

    if args.method not in ["un", "uni"]:
        print("time: %.2f, adv_acc: %.2f" % (time.time()-start_time, adv_acc*100.))

    return adv_acc



def main(args):

    # Initialize the model
    model = Model()

    seed_path = "../train/robustness_spec/seed_test_malicious/seed_feature_3416.csv"
    trie_path = "../train/robustness_spec/feature_spec/pathtrie_filled.pickle"

    feat_trie = pickle.load(open(trie_path, 'rb'))

    #x_input_test = pickle.load(f)
    x_input_test = np.genfromtxt(seed_path, delimiter=',')
    #x_input_test = np.array([item[0] for key, item in x_input_test.iteritems()])
    print("input shape:", x_input_test.shape)
    y_input_test = np.ones(x_input_test.shape[0])

    #attack_loss = model.pre_softmax[:, 0]
    attack_loss = model.xent
    benign_gradient = tf.gradients(attack_loss, model.x_input)[0]

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if args.model == "all":
            adv_accs = []
            distances = []
            samples = []
            all_is_adv = []
            if args.method in ["A", "B", "C", "D"]:
                model_names = ["baseline", "TA", "TB", "TC", "TD", "ATAB",\
                        "EAB", "ED", "RA", "RB", "RC", "RD", "RAB", "RABE", "mono"]
                if args.method == "D":
                    model_names = ["TB", "ATAB", "RD", "RAB", "RABE"]
                for model_name in model_names:
                    args.model = model_name
                    adv_acc = attack(args, saver, sess, model, benign_gradient,\
                                attack_loss, x_input_test, y_input_test, feat_trie)
                    adv_accs.append(adv_acc)
                    np.save(args.method+"_adv_accs.npy", adv_accs)
                #np.save(args.method+"_adv_accs.npy", adv_accs)
            if args.method in ["un", "uni"]:
                model_names = ["baseline", "TA", "TB", "TC", "TD", "ATAB",\
                        "EAB", "ED", "RA", "RB", "RC", "RD", "RAB", "RABE", "mono"]
                if args.method == "un":
                    model_names = ["ATAB", "RAB", "RD"]
                for model_name in model_names:
                    args.model = model_name
                    adv_acc, distance, sample, is_adv = attack(args, saver, sess, model, benign_gradient,\
                                attack_loss, x_input_test, y_input_test, feat_trie)
                    adv_accs.append(adv_acc)
                    distances.append(distance)
                    samples.append(sample)
                    all_is_adv.append(is_adv)
                    np.save(args.method+"_adv_accs_200K.npy", np.array(adv_accs))
                    np.save(args.method+"_adv_distances_200K.npy", np.array(distances))
                    np.save(args.method+"_adv_samples_200K.npy", np.array(samples))
                    np.save(args.method+"_is_adv_200K.npy", np.array(all_is_adv))

        else:
            adv_accs = attack(args, saver, sess, model, benign_gradient,\
                            attack_loss, x_input_test, y_input_test, feat_trie)

            np.save(args.method+"_"+args.model+"_adv_accs.npy", adv_accs)



if __name__=='__main__':
    args = parse_args()
    #if(args.method not in ["d2", "i", "b1", "c2"]):
        #print ("--method: d2, i, b1, c2")
        #exit(1)
    if(args.iters<=0):
        print( "--itrers: Number of iterations for gradient attacks should not be less than 0!")
        exit(1)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)
