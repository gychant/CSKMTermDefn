"""
Sample qualified candidate triples for manual evaluation.
"""
import os
import random
from tqdm import tqdm
import numpy as np
import xlsxwriter
from relation import rel2text


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
                # rel, subj, obj, _ = line.strip().split("\t")
                rel, subj, obj = line.strip().split("\t")
                subj = subj.lower()
                obj = obj.lower()
                rel = rel.lower()
                rel2triples[rel].add((rel, subj, obj))
    return rel2triples


def write_to_spreadsheet(rel2triples, output_dir, data_name):
    output_path = os.path.join(output_dir, "{}.xlsx".format(data_name))
    workbook = xlsxwriter.Workbook(output_path)

    for rel in rel2triples:
        worksheet = workbook.add_worksheet(rel)

        for i, triple in enumerate(rel2triples[rel]):
            rel, subj, obj, score_str = triple
            worksheet.write(i, 0, rel)
            worksheet.write(i, 1, subj)
            worksheet.write(i, 2, obj)
            worksheet.write(i, 3, score_str)
    workbook.close()
    print("Written triples to speadsheet {}".format(output_path))


def sample_triples(input_dir, output_dir, sampling_size, candidate_triples=None,
                   from_top_k=None, min_score=None, data_name=None,
                   output_xlsx=False):
    assert from_top_k is not None or min_score is not None
    random.seed(123)

    print("Processing {} predictions ...".format(data_name))
    test_files = [f for f in os.listdir(input_dir)
                  if os.path.isfile(os.path.join(input_dir, f))
                  and os.path.splitext(os.path.basename(f))[0] in rel2text]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rel2triples = dict()
    for test_file in test_files:
        # print("Reading file {} ...".format(test_file))
        relation = os.path.splitext(os.path.basename(test_file))[0]
        triples = []

        cnt = 0
        with open(os.path.join(input_dir, test_file)) as f:
            for line in f:
                rel, subj, obj, score_str = line.strip().split("\t")
                subj = subj.lower()
                obj = obj.lower()
                rel = rel.lower()
                score = float(score_str)
                cnt += 1

                if candidate_triples is not None \
                        and (rel, subj, obj) not in candidate_triples[rel]:
                    continue

                if (min_score is not None and score >= min_score) \
                        or (from_top_k is not None and len(triples) < from_top_k):
                    triples.append((rel, subj, obj, score_str))
                else:
                    break
        print("\nrelation:", relation)
        print("Num of Qualified:", len(triples))

        if sampling_size < len(triples):
            samples = random.sample(triples, sampling_size)
        else:
            samples = triples

        rel2triples[relation] = samples

        # Write sampled triples to file
        samples = sorted(samples, key=lambda x: float(x[3]), reverse=True)
        with open(os.path.join(output_dir, "{}_{}_samples.txt".format(
                relation, sampling_size)), "w") as f:
            for triple in samples:
                f.write("\t".join(triple))
                f.write("\n")

    if output_xlsx:
        write_to_spreadsheet(rel2triples, output_dir, data_name)
    print("DONE.")


if __name__ == "__main__":
    sampling_size = 50

    candidate_triples = read_candidate_relationwise_triples(
        "./data/wiktionary_relationwise_candidates_by_pos_tag_core")

    for rel in candidate_triples:
        print("\nrelation:", rel)
        print("Num of total candidates:", len(candidate_triples[rel]))

    print("Bilinear AVG sampling ...")
    sample_triples(input_dir="./data/bilearavg_wiktionary_core",
                   output_dir="./data/bilearavg_wiktionary_core_samples",
                   sampling_size=sampling_size, min_score=0.9, data_name="BILINEAR-AVG",
                   candidate_triples=candidate_triples,
                   output_xlsx=True)

    print("KG-BERT sampling ...")
    sample_triples(input_dir="./data/kgbert_wiktionary_core",
                   output_dir="./data/kgbert_wiktionary_core_samples",
                   sampling_size=sampling_size, min_score=0.9, data_name="KG-BERT",
                   candidate_triples=candidate_triples,
                   output_xlsx=True)

    print("PMI sampling ...")
    sample_triples(input_dir="./data/pmi_coherency_wiktionary_core",
                   output_dir="./data/pmi_coherency_wiktionary_core_samples",
                   sampling_size=sampling_size, from_top_k=1000, data_name="PMI",
                   candidate_triples=candidate_triples,
                   output_xlsx=True)

