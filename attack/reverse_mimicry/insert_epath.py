#! /usr/bin/env python
import sys
import os
import argparse
import pickle

_current_dir = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_current_dir, ".."))
sys.path.append(PROJECT_ROOT)

from mylib.config import config
evademl_dir = config.get('evademl', 'project_path')
sys.path.append(evademl_dir)

from lib.pdf_genome import PdfGenome
from lib.common import deepcopy

def parse_args():
    parser = argparse.ArgumentParser(description='Test insert exploit path to a benign PDF file.')
    parser.add_argument('--mal', type=str, help='Malicious file location.', required=True)
    parser.add_argument('--ben', type=str, help='Benign file location.', required=True)
    parser.add_argument('--var_dir', type=str, help='Variant files directory.', required=True)
    parser.add_argument('--exploit_spec', type=str, help='Exploit specification file.', required=True)
    return parser.parse_args()

def main(args):
    mal_sha1 = os.path.basename(args.mal).split('.')[0]
    # load malicious pdf file.
    mal_obj = PdfGenome.load_genome(args.mal, noxref=True)
    # load benign pdf file.
    ben_obj = PdfGenome.load_genome(args.ben, noxref=True)
    
    newpdf = deepcopy(ben_obj)
    # get exploit path from the malicious pdf file.
    exploit_spec = pickle.load(open(args.exploit_spec, 'rb'))
    epaths = exploit_spec[mal_sha1]
    
    all_ben_paths = PdfGenome.get_object_paths(ben_obj, set())
    
    # inject each path from exploit paths
    for path in epaths:
        src_path = None
        # what is the object from path? get insertable path.
        for j in xrange(1, len(path)):
            if path[:-j] in all_ben_paths:
                src_path = path[:-j]
                break
        if src_path is None:
            src_path = ['/Root']
        if j > 1:
            tgt_path = path[:-j+1]
        else:
            tgt_path = path
        PdfGenome.insert_under(newpdf, src_path, mal_obj, tgt_path)

    outname = '%s/%s_%s' % (args.var_dir, mal_sha1, os.path.basename(args.ben))
    PdfGenome.save_to_file(newpdf, outname)

if __name__=='__main__':
    args = parse_args()
    main(args)

