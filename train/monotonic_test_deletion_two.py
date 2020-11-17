#! /usr/bin/env python
import sys
import os
import argparse
import pickle
import xgboost as xgb
import scipy

def parse_args():
    parser = argparse.ArgumentParser(description='Check monotonic classifiers if lower bound of deletion intervals are classified as benign.')
    parser.add_argument('--model', type=str, help='Name of the model.', required=True)
    return parser.parse_args()

def main(args):
    # load the trained model
    model = xgb.Booster()
    modelpath = '../models/monotonic'
    model.load_model("%s/%s.bin" % (modelpath, args.model))

    # load the one deletion intervals
    test_interval_path = 'robustness_spec/seed_test_malicious/mutate_delete_two/pickles'
    # Load the test data
    print 'Loading the testing interval datasets...'
    #x_input_test = pickle.load(open(os.path.join(test_interval_path, 'x_input.pickle'), 'rb'))
    #if type(x_input_test[0]) == scipy.sparse.csr.csr_matrix:
    #    x_input_test = np.array([item.toarray()[0] for item in x_input_test])
    y_input_test = pickle.load(open(os.path.join(test_interval_path, 'y_input.pickle'), 'rb'))
    splits_test = pickle.load(open(os.path.join(test_interval_path, 'splits.pickle'), 'rb'))
    vectors_all_test = pickle.load(open(os.path.join(test_interval_path, 'vectors_all.pickle'), "rb"))

    print vectors_all_test.shape
    print type(vectors_all_test)

    dtest = xgb.DMatrix(vectors_all_test.tolist())
    y_pred = model.predict(dtest)

    total = 0
    incorrect = 0
    j = 0
    for cur_cnt in splits_test:
        # everything in y_pred[j:j+cur_cnt] has to be one
        total += 1
        #print y_pred[j:j+cur_cnt]
        for item in y_pred[j:j+cur_cnt]:
            if item < 0.5:
                incorrect += 1
                break
        j += cur_cnt
    print 'total:', total
    print 'incorrect:', incorrect
    #print 'ERA:', cnt/float(total)



if __name__=='__main__':
    args = parse_args()
    main(args)
