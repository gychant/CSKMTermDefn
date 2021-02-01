"""
Extracting commonsense knowledge triples using frequent POS tag patterns.
"""

import logging
import os
import pickle
import json
import re
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WiktionaryTripleExtractor(object):
    def __init__(self, included_relations=[], excluded_relations=[],
                 pos_seq_counter_file=None, strategy="pos_tag", relationwise=False,
                 topk_pos_patterns=15, max_gram_num=5, min_score=0.9):
        """
        :param included_relations:
        :param excluded_relations:
        :param strategy: ["pos_tag", "ngram"]
        :param max_gram_num:
        :param min_score:
        """
        assert strategy in ["pos_tag", "ngram"]

        self.included_relations = set([r.lower() for r in included_relations])
        self.excluded_relations = set([r.lower() for r in excluded_relations])
        self.strategy = strategy
        self.topk_pos_patterns = topk_pos_patterns
        self.max_gram_num = max_gram_num
        self.min_score = min_score
        self.relationwise = relationwise
        self.stop_words = set(stopwords.words('english'))

        if self.strategy == "pos_tag":
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            if pos_seq_counter_file is not None:
                with open(pos_seq_counter_file, "rb") as f:
                    if relationwise:
                        counter_pos_seq = pickle.load(f)
                        self.freq_pos_patterns = {
                            rel: dict(Counter(counter_pos_seq[rel]).most_common(topk_pos_patterns))
                            for rel in counter_pos_seq
                            if rel in self.included_relations and rel not in self.excluded_relations}
                        logger.info("Top {} patterns: {}".format(
                            topk_pos_patterns, self.freq_pos_patterns))
                    else:
                        counter_pos_seq = Counter(pickle.load(f))
                        logger.info("Num of patterns: {}".format(len(counter_pos_seq)))
                        self.freq_pos_patterns = dict(counter_pos_seq.most_common(topk_pos_patterns))
                        logger.info("Top {} patterns: {}".format(
                            topk_pos_patterns, self.freq_pos_patterns))

    def _build_hyphen_dict(self, text):
        # group tokens connected by hyphens
        hyphen_dict = {}
        for token in text.split():
            # remove period
            if token.endswith("."):
                token = token[:-1]

            if "-" in token:
                for tok in token.split("-"):
                    hyphen_dict[tok] = token
        return hyphen_dict

    def _get_matched_relation(self, pos_tag_ngram):
        """
        Find the relations that the input pos tag ngram belongs to.
        :param pos_tag_ngram:
        :return:
        """
        relations = []
        for rel in self.freq_pos_patterns:
            if tuple(pos_tag_ngram) in self.freq_pos_patterns[rel]:
                relations.append(rel)
        return relations

    def _generate_candidate_by_pos_tag(self, defn):
        hyphen_dict = self._build_hyphen_dict(defn)
        tokens, pos_tags = zip(*[(token.text, token.pos_) for token in self.nlp(defn)])
        for ngram, pos_tag_ngram in zip(self._ngram_generator(tokens, max_n=self.max_gram_num),
                                        self._ngram_generator(pos_tags, max_n=self.max_gram_num)):
            if self.relationwise:
                relations = self._get_matched_relation(pos_tag_ngram)
                if len(relations) == 0:
                    continue

            if not self.relationwise and tuple(pos_tag_ngram) not in self.freq_pos_patterns:
                continue

            candidate = " ".join(ngram)
            if candidate in self.stop_words:
                continue

            # if candidate is part of phrases with hyphens, recover it
            for token in hyphen_dict:
                if token not in candidate:
                    continue

                new_candidate = candidate.replace(token, hyphen_dict[token])
                if new_candidate in defn:
                    candidate = new_candidate
                    break

            if self.relationwise:
                yield candidate, relations
            else:
                yield candidate

    def extract_candidates(self, input_file_path, output_file_path,
                           vocab_file_path=None):
        """
        :param input_file_path:
        :param output_file_path:
        :param vocab_file_path:
        :return:
        """
        if not os.path.exists(input_file_path):
            logger.error("File {} does not exist ...".format(input_file_path))
            return

        if vocab_file_path is not None:
            logger.info("Loading vocab from {} ...".format(vocab_file_path))
            vocab = set()
            with open(vocab_file_path) as f:
                for line in f:
                    vocab.add(line.strip().replace("_", " "))
            logger.info("Vocab size: {}".format(len(vocab)))

        num_candidates = 0
        subject_set = set()
        f_out = open(output_file_path, "w")
        with open(input_file_path, "r") as f:
            for line in tqdm(f):
                data = json.loads(line.strip())
                if "noun" not in data["defn"] \
                        or len(data["defn"]["noun"].strip()) == 0:
                    continue

                subj = data["text"].replace("_", " ")
                defn = data["defn"]["noun"]
                triples = set()

                # ignore plural form definitions, etc
                if "plural of" in defn.lower() \
                        or "alternative form of" in defn.lower() \
                        or "alternative spelling of" in defn.lower() \
                        or "misspelling of" in defn.lower():
                    continue

                if vocab_file_path is not None and subj not in vocab:
                    continue

                subject_set.add(subj)

                if self.relationwise:
                    for candidate, relations in self._generate_candidate_by_pos_tag(defn):
                        for rel in relations:
                            if candidate != subj:
                                triples.add((rel, subj, candidate))
                else:
                    for candidate in self._generate_candidate_by_pos_tag(defn):
                        if candidate != subj:
                            triples.add((subj, candidate))

                if self.relationwise:
                    for rel, head, tail in triples:
                        f_out.write(rel + "\t" + head + "\t" + tail + "\n")
                        num_candidates += 1
                else:
                    for head, tail in triples:
                        f_out.write(head + "\t" + tail + "\n")
                        num_candidates += 1

        f_out.close()
        logger.info("Num of term definitions: {}".format(len(subject_set)))
        logger.info("Extracted {} candidates.".format(num_candidates))
        logger.info("Written candidates to {}".format(output_file_path))

    def split_triples_by_relation(self, input_file_path, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        opened_files = {}
        with open(input_file_path) as f:
            for line in tqdm(f):
                rel, _, _ = line.strip().split("\t")
                tgt_file_path = os.path.join(output_dir, "{}.txt".format(rel))
                if rel not in opened_files:
                    rel_file = open(tgt_file_path, "w")
                    opened_files[rel] = rel_file
                else:
                    rel_file = opened_files[rel]
                rel_file.write(line)

        for _, f in opened_files.items():
            f.close()
        logger.info("Splitting DONE.")

    @staticmethod
    def _ngram_generator(tokens, max_n=5):
        for n in range(1, max_n + 1):
            yield from ngrams(tokens, n)


if __name__ == "__main__":
    import time
    start_time = time.time()
    extractor = WiktionaryTripleExtractor(
        included_relations=["isa", "capableof", "usedfor",
                            "causesdesire", "receivesaction",
                            "atlocation", "partof",
                            "motivatedbygoal", "hassubevent",
                            "causes", "madeof", "desires",
                            "createdby", "hasproperty"],
        excluded_relations=["definedas"],
        pos_seq_counter_file="./data/conceptnet_relationwise_pos_seq_counter.pkl",
        relationwise=True, strategy="pos_tag", topk_pos_patterns=15,
        max_gram_num=8, min_score=0.3)

    extractor.extract_candidates(
        input_file_path="./data/conceptnet/conceptnet_vocab_definition.jsonl",
        output_file_path="./data/wiktionary_relationwise_candidates_by_pos_tag_core.txt",
        vocab_file_path="./data/conceptnet/conceptnet_vocab_pos_tag_core.tsv")
    logger.info("extract_candidates DONE.")

    extractor.split_triples_by_relation(
        input_file_path="./data/wiktionary_relationwise_candidates_by_pos_tag_core.txt",
        output_dir="./data/wiktionary_relationwise_candidates_by_pos_tag_core")

    logger.info("split_triples_by_relation DONE.")
    logger.info("Time cost: {} secs".format(time.time() - start_time))

