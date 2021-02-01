"""
Merge split files that are created for parallel predictions.
"""
import os
import json
from tqdm import tqdm
from relation import rel2text


def merge_triples(input_dir, output_dir):
    test_files = [f for f in os.listdir(input_dir)
                  if os.path.isfile(os.path.join(input_dir, f))]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for rel in rel2text:
        triples = []
        for test_file in test_files:
            file_name = os.path.splitext(os.path.basename(test_file))[0]
            relation = file_name.split("_")[0]

            if rel != relation:
                continue

            with open(os.path.join(input_dir, test_file)) as f:
                for line in f:
                    rel, subj, obj, score_str = line.strip().split("\t")
                    subj = subj.lower()
                    obj = obj.lower()
                    rel = rel.lower()
                    triples.append((rel, subj, obj, score_str))

        if len(triples) == 0:
            continue

        print("rel:", rel)
        print("Num of triples:", len(triples))
        # Write triples to file
        triples = sorted(triples, key=lambda x: float(x[3]), reverse=True)
        with open(os.path.join(output_dir, "{}.txt".format(rel)), "w") as f:
            for triple in triples:
                f.write("\t".join(triple))
                f.write("\n")
    print("DONE.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='input_dir')
    parser.add_argument('--output_dir', type=str, help='output_dir')
    args = parser.parse_args()

    merge_triples(input_dir="../data/pmi_coherency_wiktionary_core_split",
                  output_dir="../data/pmi_coherency_wiktionary_core")

