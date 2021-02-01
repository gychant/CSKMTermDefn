"""
Analyze part-of-speech tag sequence of triple subjects and objects
"""

import logging
import os
import pickle
import json
import re
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter
from tqdm import tqdm
import spacy

from wiktionary.ckbc_triple_scorer import CkbcTripleScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")


class TriplePosTagAnalyzer(object):
    def __init__(self, included_relations=[], excluded_relations=[]):
        self.included_relations = set([r.lower() for r in included_relations])
        self.excluded_relations = set([r.lower() for r in excluded_relations])

    def scan(self, input_file_path, output_file_path,
             relationwise=True, min_seq_len=0):
        """
        Scan the triple file, get the part-of-speech tags of multi-word concepts,
        and add to the counter.
        :param input_file_path:
        :param output_file_path:
        :param relationwise:
        :param min_seq_len:
        :return:
        """
        if not os.path.exists(input_file_path):
            logger.error("File {} does not exist ...".format(input_file_path))
            return

        if relationwise:
            counter_pos_seq = dict()
        else:
            counter_pos_seq = Counter()

        with open(input_file_path, "r") as f:
            for line in tqdm(f):
                data = json.loads(line.strip())

                subj = data["head"].replace("_", " ")
                obj = data["tail"].replace("_", " ")
                rel = data["rel"]
                subj_tokens = subj.strip().split()
                obj_tokens = obj.strip().split()

                if len(subj_tokens) > min_seq_len:
                    pos_tags = tuple([token.pos_ for token in nlp(subj)])
                    if relationwise:
                        if rel not in counter_pos_seq:
                            counter_pos_seq[rel] = Counter()
                        counter_pos_seq[rel][pos_tags] += 1
                    else:
                        counter_pos_seq[pos_tags] += 1

                if len(obj_tokens) > min_seq_len and rel not in self.excluded_relations:
                    pos_tags = tuple([token.pos_ for token in nlp(obj)])
                    if relationwise:
                        if rel not in counter_pos_seq:
                            counter_pos_seq[rel] = Counter()
                        counter_pos_seq[rel][pos_tags] += 1
                    else:
                        counter_pos_seq[pos_tags] += 1

        with open(output_file_path, "wb") as f_out:
            pickle.dump(dict(counter_pos_seq), f_out)
        logger.info(dict(counter_pos_seq))
        if not relationwise:
            logger.info("Num of patterns: {}".format(len(dict(counter_pos_seq))))
        logger.info("Saved counter to {}".format(output_file_path))
        logger.info("Scan DONE.")


if __name__ == "__main__":
    import time
    start_time = time.time()
    analyzer = TriplePosTagAnalyzer(
        included_relations=["isa", "capableof", "usedfor",
                            "causesdesire", "receivesaction",
                            "atlocation", "partof",
                            "motivatedbygoal", "hassubevent",
                            "causes", "madeof", "desires",
                            "createdby", "hasproperty"],
        excluded_relations=["relatedto"])
    analyzer.scan("./data/conceptnet/conceptnet_en.jsonl",
                  "./data/conceptnet_relationwise_pos_seq_counter.pkl",
                  relationwise=True, min_seq_len=0)
    logger.info("Time cost: {} secs".format(time.time() - start_time))

