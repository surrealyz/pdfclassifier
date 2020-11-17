#! /usr/bin/env python
import os
import sys
import argparse
import numpy as np
from numpy import genfromtxt
import pickle
import re
import pdfrw
from collections import defaultdict

_current_dir = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_current_dir, ".."))
sys.path.append(PROJECT_ROOT)

from mylib.config import config
evademl_dir = config.get('evademl', 'project_path')
sys.path.append(evademl_dir)

from lib.pdf_genome import PdfGenome
from lib.common import deepcopy

# prepare all regex and the substitution
all_matches = [
        (re.compile(r"/Resources/(ExtGState|ColorSpace|Pattern|Shading|XObject|Font|Properties|Para)/[^/]+"), "/Resources/\\1/Name"),
        (re.compile(r"^Pages/(Kids/|Parent/)*(Kids$|Kids/|Parent/|Parent$)"), "Pages/"),
        (re.compile(r"/(Kids/|Parent/)*(Kids$|Kids/|Parent/|Parent$)"), "/"),
        (re.compile(r"(Prev/|Next/|First/|Last/)+"), ""),
        (re.compile(r"^Names/(Dests|AP|JavaScript|Pages|Templates|IDS|URLS|EmbeddedFiles|AlternatePresentations|Renditions)/(Kids/|Parent/)*Names"), "Names/\\1/Names"),
        (re.compile(r"^StructTreeRoot/IDTree/(Kids/)*Names"), "StructTreeRoot/IDTree/Names"),
        (re.compile(r"^(StructTreeRoot/ParentTree|PageLabels)/(Kids/|Parent/)+(Nums|Limits)"), "\\1/\\3"),
        (re.compile(r"^StructTreeRoot/ParentTree/Nums/(K/|P/)+"), "StructTreeRoot/ParentTree/Nums/"),
        (re.compile(r"^(StructTreeRoot|Outlines/SE)/(RoleMap|ClassMap)/[^/]+"), "\\1/\\2/Name"),
        (re.compile(r"^(StructTreeRoot|Outlines/SE)/(K/|P/)*"), "\\1/"),
        (re.compile(r"^(Extensions|Dests)/[^/]+"), "\\1/Name"),
        (re.compile(r"Font/([^/]+)/CharProcs/[^/]+"), "Font/\\1/CharProcs/Name"),
        (re.compile(r"^(AcroForm/(Fields/|C0/)?DR/)(ExtGState|ColorSpace|Pattern|Shading|XObject|Font|Properties)/[^/]+"), "\\1\\3/Name"),
        (re.compile(r"/AP/(D|N)/[^/]+"), "/AP/\\1/Name"),
        (re.compile(r"Threads/F/(V/|N/)*"), "Threads/F"),
        (re.compile(r"^(StructTreeRoot|Outlines/SE)/Info/[^/]+"), "\\1/Info/Name"),
        (re.compile(r"ColorSpace/([^/]+)/Colorants/[^/]+"), "ColorSpace/\\1/Colorants/Name"),
        (re.compile(r"ColorSpace/Colorants/[^/]+"), "ColorSpace/Colorants/Name"),
        (re.compile(r"Collection/Schema/[^/]+"), "Collection/Schema/Name"),
]

def compact(path):
    for p, newpattern in all_matches:
        path = p.sub(newpattern, path)
    return path

def parse_args():
    parser = argparse.ArgumentParser(description='Generate PDFs from evasive feature vectors.')
    return parser.parse_args()

def build_genome_dict():
    global genome_dict
    global path_to_idx
    global idx_to_path
    genome_dict = {}
    path_to_idx = {}
    idx_to_path = {}

    with open('robustness_spec/features.nppf', 'r') as fin:
        header = True
        idx = 0
        for line in fin:
            if header is True:
                header = False
                continue
            key = line.rstrip().replace('\x00', '/').rstrip('/')
            path_to_idx[key] = idx
            idx_to_path[idx] = key
            idx += 1

    # make a dictionary from feature index to the PDF and pdfrw path
    fname = 'robustness_spec/minimal_obj_compact_v3_true.txt'
    with open(fname, 'r') as f:
        for line in f:
            # StructTreeRoot/Pg/Resources/XObject/Name/Resources/ExtGState/Name/op    3   p4298.pdf   ['/StructTreeRoot/K/Pg/Resources/XObject/Fm0/Resources/ExtGState/GS0/op', '/StructTreeRoot/K/Pg/Resources/XObject/Fm2/Resources/ExtGState/GS0/op', '/StructTreeRoot/K/Pg/Resources/XObject/Fm3/Resources/ExtGState/GS0/op']
            shortpath, count, train_f, fullpaths = line.rstrip().split('\t')
            idx = path_to_idx[shortpath]
            genome_dict[idx] = (train_f, eval(fullpaths))

    return

def get_ins_del(seed_vec, vec):
    ins_indices = [] # 0 -> 1
    del_indices = [] # 1 -> 0
    for k in range(3514):
        if seed_vec[k] < vec[k]:
            ins_indices.append(k)
        if seed_vec[k] > vec[k]:
            del_indices.append(k)
    return ins_indices, del_indices

# generate newpdf from src_entry
def generate_pdf(src_entry, sha1, ins_indices, del_indices, model_name):
    global genome_dict
    global idx_to_path
    # deep copy
    newpdf = deepcopy(src_entry)

    ### INSERTION
    for index in ins_indices:
        # find the newobj
        try:
            train_f, fullpaths = genome_dict[index]
        except KeyError:
            continue
        fname = '../data/traintest_all_500test/train_benign/%s' % train_f
        try:
            tgt_entry = PdfGenome.load_genome(fname, noxref = True)
        except pdfrw.errors.PdfParseError:
            tgt_entry = PdfGenome.load_genome(fname, noxref = False)

        # do deterministic
        tgt_path = ['/'+item for item in ('/Root' + fullpaths[0]).split('/')[1:]]
        #tgt_parent, tgt_key = PdfGenome.get_parent_key(tgt_entry, tgt_path)

        # find the longest prefix that exists in src_entry
        #parent = newpdf
        #for i in range(len(tgt_path)-1, 0, -1):
        #    key = tgt_path[:i]
        src_parent = newpdf
        i = 0
        for key in tgt_path[:-1]:
            try:
                src_parent = src_parent[key]
                i += 1
            except (KeyError, TypeError):
                #print tgt_path
                #print sha1
                #print index
                #print cur_iter
                #raise SystemExit
                break
        #key = tgt_path[i-1:i]
        # last parent should work
        src_key = tgt_path[:i]
        if src_key != ['/Root']:
            tgt_key = tgt_path[:i]
            #print src_key
            #print tgt_key
            try:
                PdfGenome.insert(newpdf, src_key, tgt_entry, tgt_key)
            except Exception:
                pass
        else:
            tgt_key = tgt_path[:i+1]
            #print src_key
            #print tgt_key
            # do a insert_under
            PdfGenome.insert_under(newpdf, src_key, tgt_entry, tgt_key)

    ### DELETION
    # for each compact path, I need a set of original paths from the PDF. then I need to delete all of them.
    # get the compact path to path mapping
    compact_to_full = defaultdict(list)
    paths = PdfGenome.get_object_paths(src_entry)
    for ext_id in range(len(paths)):
        fullpath = paths[ext_id]
        fullkey = ''.join([item for item in fullpath[1:] if type(item) != int])
        # IMPORTANT: make this path compact
        key = compact(fullkey[1:])
        compact_to_full[key].append(fullpath)

    # TODO: remove debug
    print compact_to_full

    for index in del_indices:
        compactpath = idx_to_path[index]
        for path in compact_to_full[compactpath]:
            # TODO: remove debug
            print 'delete:', path
            # delete the full path
            try:
                PdfGenome.delete(newpdf, path)
            except Exception:
                # the parent may already be deleted
                continue


    file_dir = 'unrestricted/%s' % model_name
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    pdf_path = '%s/%s.pdf' % (file_dir, sha1)
    PdfGenome.save_to_file(newpdf, pdf_path)
    return newpdf, pdf_path


def main(args):
    global genome_dict
    global idx_to_path
    build_genome_dict()

    # load the npy file
    adv_samples = np.load('../data/un_adv_samples.npy')

    # load the seed feature vectors
    seed_dict = pickle.load(open('robustness_spec/seed_test_malicious/feat_dict_3416.pickle', 'rb'))
    seed_features = genfromtxt('robustness_spec/seed_test_malicious/seed_feature_3416.csv', delimiter=',')
    # load the seed entries together. deepcopy later
    all_sha1 = seed_dict.keys()
    sha1_500 = [item.split('.')[0] for item in os.listdir('../data/500_seed_pdfs/')]
    v_i_to_sha1 = {}
    for i in range(len(all_sha1)):
        if all_sha1[i] in sha1_500:
            v_i_to_sha1[i] = all_sha1[i]

    # each of the 15 models
    # "baseline", "TA", "TB", "TC", "TD", "ATAB", "EAB", "ED", "RA", "RB", "RC", "RD", "RAB", "RABE", "mono"
    model_names = ['baseline', 'adv_a', 'adv_b', 'adv_c', 'adv_d', 'adv_ab', 'ensemble_ab', 'ensemble_d', 'robust_a', 'robust_b', 'robust_c', 'robust_d', 'robust_ab', 'robust_abe', 'robust_e']

    #for m_i in range(15):
    for m_i in range(8, 15):
        # each of the 3416 evasive vectors against the model
        res = adv_samples[m_i]
        for v_i in range(3416):
            # figure out the difference of this vector with the original feature vector
            if v_i not in v_i_to_sha1.keys():
                continue
            vector = res[v_i]
            seed_vec = seed_features[v_i]
            # get the difference
            # all the insertion indices
            # all the deletion indices
            ins_indices, del_indices = get_ins_del(seed_vec, vector)
            # get the original PDF object, then mutate.
            sha1 = v_i_to_sha1[v_i]
            src_entry = PdfGenome.load_genome('../data/500_seed_pdfs/%s.pdf' % sha1, noxref = True)
            generate_pdf(src_entry, all_sha1[v_i], ins_indices, del_indices, model_names[m_i])

    return


if __name__=='__main__':
    args = parse_args()
    main(args)
