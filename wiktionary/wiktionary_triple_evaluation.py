"""
Assign plausibility scores to triples extracted from Wiktionary with BiLinear AVG model.
"""
from __future__ import absolute_import, division, print_function

import os
import logging
from tqdm import tqdm

from relation import rel2text
from ckbc_triple_scorer import CkbcTripleScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_scorer(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scorer = CkbcTripleScorer()

    # run prediction on all the files under the data_dir
    test_files = [f for f in os.listdir(input_dir)
                  if os.path.isfile(os.path.join(input_dir, f))
                  and os.path.splitext(os.path.basename(f))[0] in rel2text]

    for test_file in test_files:
        logger.info("Evaluating file {} ...".format(test_file))

        scored_samples = []
        cnt_miss_score = 0
        with open(os.path.join(input_dir, test_file)) as f:
            for line in tqdm(f):
                line = line.strip()
                rel, head, tail = line.split("\t")
                result = scorer.score_triple(head, tail, "all")
                rel2score = dict(result)
                if rel in rel2score:
                    score = rel2score[rel]
                else:
                    score = 0
                    cnt_miss_score += 1
                scored_samples.append((rel, head, tail, score))

        # sort triples by scores in descending order and
        # write triples and the scores to new files
        scored_samples.sort(key=lambda x: x[3], reverse=True)

        output_file = os.path.join(output_dir, test_file)
        with open(output_file, "w") as writer:
            for spl in scored_samples:
                writer.write("\t".join(spl[:3]) + "\t" + "{:.5f}".format(spl[3]))
                writer.write("\n")
        print("cnt_miss_score:", cnt_miss_score)
        logger.info("Written to file {}".format(output_file))


if __name__ == "__main__":
    run_scorer(input_dir="./data/wiktionary_relationwise_candidates_by_pos_tag_core",
               output_dir="./data/bilearavg_wiktionary_core")

