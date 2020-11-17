#! /usr/bin/env python
import os
import sys
import pickle

to_skip = ['1ec657f52bf1811af14d7da549cb6add70c778f0', 'b01be494ac00843796cb200caf91e7ab2e997c34', 'b4f13bf5f4174fd7a7c2a52b21309da8da0b33ce', 'f2a9170030b999834018203f468ea9bcf8e444c0', 'f3efb335a617ecb76e1e7519bc7c2c3df8fa47f6']

def main(argv):
    # get seed sha1s.
    seed_paths = pickle.load(open('../data/shuffled_seed_paths_most_benign.pickle', 'rb'))
    # load exploit paths pickle.
    exploit_spec = pickle.load(open('../data/traintest_all_500test/exploit_spec/test_malicious.pickle', 'rb'))
    cnt = 0
    for fname in seed_paths:
        sha1 = os.path.basename(fname).split('.')[0]
        paths = exploit_spec.get(sha1, None)
        # insert these
        if paths is not None and sha1 not in to_skip:
            cmd = 'python insert_epath.py --mal ../data/traintest_all_500test/test_malicious/%s.pdf --ben ../../data/most_benign_genome/win08.pdf --var_dir ./test_files --exploit_spec ../../data/traintest_all_500test/exploit_spec/test_malicious.pickle' % sha1
            print cmd
            os.system(cmd)
            cnt += 1
    print cnt


if __name__=='__main__':
    main(sys.argv)
