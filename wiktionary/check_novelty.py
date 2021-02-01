"""
Check the novelty of extracted triples.
"""

import os
import json
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

from relation import rel2text

remove_stop_words = True
use_bag_of_words = True
use_lemma = True

if remove_stop_words:
    stop_word_set = set(stopwords.words('english'))
if use_lemma:
    lemmatizer = WordNetLemmatizer()
    stemmer = LancasterStemmer()


def transform(concept):
    tokens = [tok for tok in concept.split()]
    if remove_stop_words:
        tokens = [tok for tok in tokens if tok not in stop_word_set]

    if use_lemma:
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
        tokens = [stemmer.stem(tok) for tok in tokens]

    if use_bag_of_words:
        tokens = sorted(list(set(tokens)))
    return " ".join(tokens)


def read_candidate_relationwise_triples(input_dir, suffix=""):
    rel2triples = dict()

    suffix_offset = -len(suffix) if len(suffix) > 0 else None
    test_files = [f for f in os.listdir(input_dir)
                  if os.path.isfile(os.path.join(input_dir, f))
                  and os.path.splitext(os.path.basename(f))[0][:suffix_offset] in rel2text]

    for test_file in test_files:
        print("Reading file {} ...".format(test_file))
        relation = os.path.splitext(os.path.basename(test_file))[0][:suffix_offset]
        rel2triples[relation] = set()

        with open(os.path.join(input_dir, test_file)) as f:
            for line in tqdm(f):
                rel, subj, obj, _ = line.strip().split("\t")
                subj = subj.lower()
                obj = obj.lower()
                rel = rel.lower()
                rel2triples[rel].add((rel, subj, obj))
    return rel2triples


def read_conceptnet_relationwise_triples(file_path):
    rel2triples = dict()
    with open(file_path) as f:
        for line in tqdm(f):
            data = json.loads(line.strip())

            subj = transform(data["head"].replace("_", " ").lower())
            obj = transform(data["tail"].replace("_", " ").lower())
            rel = data["rel"].lower()

            if rel not in rel2triples:
                rel2triples[rel] = set()
            rel2triples[rel].add((rel, subj, obj))
    return rel2triples


def read_ckbc_relationwise_triples(file_path):
    rel2triples = dict()
    with open(file_path) as f:
        for line in tqdm(f):
            rel, subj, obj, _ = line.strip().split("\t")
            subj = transform(subj.lower())
            obj = transform(obj.lower())
            rel = rel.lower()

            if rel not in rel2triples:
                rel2triples[rel] = set()
            rel2triples[rel].add((rel, subj, obj))
    return rel2triples


def read_spreadsheet(input_dir, data_name):
    rel2triples = dict()
    xl = pd.ExcelFile(os.path.join(input_dir, "{}.xlsx".format(data_name)),
                      engine="openpyxl")
    relations = xl.sheet_names

    for rel in relations:
        print("rel:", rel)
        rel2triples[rel] = set()
        df = xl.parse(rel, header=None)
        # print(df)
        for index, row in df.iterrows():
            rel, subj, obj, score_str = row
            subj = transform(subj.lower())
            obj = transform(obj.lower())
            rel = rel.lower()
            rel2triples[rel].add((rel, subj, obj))
    xl.close()
    return rel2triples


def read_annoated_spreadsheet(input_dir, data_name):
    rel2triples = dict()
    annotated_triples = dict()

    xl = pd.ExcelFile(os.path.join(input_dir, "{}.xlsx".format(data_name)),
                      engine="openpyxl")
    relations = xl.sheet_names

    for rel in relations:
        rel2triples[rel] = set()
        df = xl.parse(rel, header=None, parse_cols="A:E")
        for index, row in df.iterrows():
            if index == 50:
                break

            rel, subj, obj, score, annotation = row[:5]
            subj = subj.lower()
            obj = obj.lower()
            rel = rel.lower()
            rel2triples[rel].add((rel, subj, obj))
            annotated_triples[(rel, subj, obj)] = annotation
    xl.close()
    return rel2triples, annotated_triples


def compute_novelty_rate(src_triples, tgt_triples, relation,
                         annotated_triples=None):
    num_novel = 0
    num_valid_novel = 0
    # print("src_triples:", list(src_triples)[:10])
    # print("tgt_triples:", list(tgt_triples)[:10])
    # input()
    for triple in tgt_triples:
        if annotated_triples:
            is_true = annotated_triples[triple]

        rel, subj, obj = triple
        subj = transform(subj.lower())
        obj = transform(obj.lower())
        rel = rel.lower()
        if (rel, subj, obj) not in src_triples:
            num_novel += 1

            if annotated_triples and is_true == 1:
                num_valid_novel += 1
                print("Novel Triple:", triple)

    novel_rate = num_novel / len(tgt_triples)
    print("\nNovelty rate of {}: {}".format(relation, novel_rate))

    if annotated_triples:
        valid_novel_rate = num_valid_novel / len(tgt_triples)
        print("Valid & Novelty rate of {}: {}".format(relation, valid_novel_rate))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=["BILINEAR-AVG", "KG-BERT", "PMI"],
                        help='model to be checked')
    parser.add_argument('--samples_only', action='store_true', help="samples_only")
    args = parser.parse_args()

    annotated_triples = None

    if args.model == "BILINEAR-AVG":
        if args.samples_only:
            candidate_triples, annotated_triples = read_annoated_spreadsheet(
                "./data/CSKGMining-AnnotatedSamples", data_name="BILINEAR-AVG")
        else:
            candidate_triples = read_candidate_relationwise_triples("./data/bilearavg_wiktionary_core")
    elif args.model == "KG-BERT":
        if args.samples_only:
            candidate_triples, annotated_triples = read_annoated_spreadsheet(
                "./data/CSKGMining-AnnotatedSamples", data_name="KG-BERT")
        else:
            candidate_triples = read_candidate_relationwise_triples("./data/kgbert_wiktionary_core")
    elif args.model == "PMI":
        if args.samples_only:
            candidate_triples, annotated_triples = read_annoated_spreadsheet(
                "./data/CSKGMining-AnnotatedSamples", data_name="PMI")
        else:
            candidate_triples = read_candidate_relationwise_triples("./data/pmi_coherency_wiktionary_core")

    relations = candidate_triples.keys()

    print("\nReading CKBC training set ...")
    ckbc_train_triples = read_ckbc_relationwise_triples("./data/CKBC/train100k.txt")

    print("\nReading ConceptNet Core ...")
    conceptnet_core_triples = read_conceptnet_relationwise_triples("./data/conceptnet/conceptnet_en_core.jsonl")

    print("\nReading ConceptNet English ...")
    conceptnet_triples = read_conceptnet_relationwise_triples("./data/conceptnet/conceptnet_en.jsonl")

    print("\nEvaluating novelty w.r.t. CKBC training set ...")
    for rel in relations:
        compute_novelty_rate(ckbc_train_triples[rel], candidate_triples[rel], rel,
                             annotated_triples)

    print("\nEvaluating novelty w.r.t. ConceptNet Core ...")
    for rel in relations:
        compute_novelty_rate(conceptnet_core_triples[rel], candidate_triples[rel], rel,
                             annotated_triples)

    print("\nEvaluating novelty w.r.t. ConceptNet English ...")
    for rel in relations:
        compute_novelty_rate(conceptnet_triples[rel], candidate_triples[rel], rel,
                             annotated_triples)
    print("DONE.")

