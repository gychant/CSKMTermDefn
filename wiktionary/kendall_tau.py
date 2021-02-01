"""
Compute the Kendall’s tau coef-ficients between pairs of models.
"""

import os
import json
from tqdm import tqdm
import numpy as np
from scipy import stats
from relation import rel2text


def remove_triple(triple):
    rel, subj, obj = triple
    if "plural of" in obj \
            or "alternative form of" in obj \
            or "alternative spelling of" in obj \
            or "misspelling of" in obj \
            or subj == obj:
        return True
    return False


def read_candidate_relationwise_triples(input_dir):
    rel2triple_idx = dict()

    test_files = [f for f in os.listdir(input_dir)
                  if os.path.isfile(os.path.join(input_dir, f))
                  and os.path.splitext(os.path.basename(f))[0] in rel2text]

    for test_file in test_files:
        print("Reading file {} ...".format(test_file))
        relation = os.path.splitext(os.path.basename(test_file))[0]
        rel2triple_idx[relation] = dict()

        with open(os.path.join(input_dir, test_file)) as f:
            for i, line in tqdm(enumerate(f)):
                rel, subj, obj = line.strip().split("\t")
                subj = subj.lower()
                obj = obj.lower()
                rel = rel.lower()
                rel2triple_idx[rel][(rel, subj, obj)] = i + 1
    return rel2triple_idx


def read_relationwise_triple_ranking(input_dir, rel2triple_idx):
    print("Reading DIR {}:".format(input_dir))
    rel2ranking = dict()

    test_files = [f for f in os.listdir(input_dir)
                  if os.path.isfile(os.path.join(input_dir, f))
                  and os.path.splitext(os.path.basename(f))[0] in rel2text]

    for test_file in test_files:
        print("Reading file {} ...".format(test_file))
        relation = os.path.splitext(os.path.basename(test_file))[0]
        rel2ranking[relation] = []

        with open(os.path.join(input_dir, test_file)) as f:
            for i, line in tqdm(enumerate(f)):
                rel, subj, obj, _ = line.strip().split("\t")
                subj = subj.lower()
                obj = obj.lower()
                rel = rel.lower()
                if (rel, subj, obj) in rel2triple_idx[rel]:
                    rel2ranking[rel].append(rel2triple_idx[rel][(rel, subj, obj)])
    return rel2ranking


if __name__ == "__main__":
    rel2triple_idx = read_candidate_relationwise_triples(
        "./data/wiktionary_relationwise_candidates_by_pos_tag_core")

    relations = rel2triple_idx.keys()

    bilinear_avg_ranking = read_relationwise_triple_ranking(
        input_dir="./data/bilearavg_wiktionary_core",
        rel2triple_idx=rel2triple_idx)

    kgbert_ranking = read_relationwise_triple_ranking(
        input_dir="./data/kgbert_wiktionary_core",
        rel2triple_idx=rel2triple_idx)

    pmi_ranking = read_relationwise_triple_ranking(
        input_dir="./data/pmi_coherency_wiktionary_core",
        rel2triple_idx=rel2triple_idx)

    print("\nEvaluating BiLinear-AVG vs. KG-BERT")
    for rel in relations:
        assert len(bilinear_avg_ranking[rel]) == len(kgbert_ranking[rel])
        tau, p_value = stats.kendalltau(bilinear_avg_ranking[rel], kgbert_ranking[rel])
        print("{}: tau:{}, p_value:{}".format(rel, tau, p_value))

    print("\nEvaluating KG-BERT vs. PMI")
    for rel in relations:
        assert len(kgbert_ranking[rel]) == len(pmi_ranking[rel])
        tau, p_value = stats.kendalltau(kgbert_ranking[rel], pmi_ranking[rel])
        print("{}: tau:{}, p_value:{}".format(rel, tau, p_value))

    print("\nEvaluating BiLinear-AVG vs. PMI")
    for rel in relations:
        assert len(bilinear_avg_ranking[rel]) == len(pmi_ranking[rel])
        tau, p_value = stats.kendalltau(bilinear_avg_ranking[rel], pmi_ranking[rel])
        print("{}: tau:{}, p_value:{}".format(rel, tau, p_value))
    print("DONE.")

