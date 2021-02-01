"""
Wrapper module of the CKBC triple scorer.
"""
from __future__ import absolute_import, division, print_function

import logging
import pickle
from ckbc_bilinear.demo_bilinear import score

logger = logging.getLogger(__name__)


class CkbcTripleScorer(object):
    def __init__(self):
        model_file_path = "./data/CKBC/ckbc-demo/Bilinear.pickle"
        model = pickle.load(open(model_file_path, "rb"), encoding="latin1")
        self.Rel = model['rel']
        self.We = model['embeddings']
        self.Weight = model['weight']
        self.Offset = model['bias']
        self.words = model['words_name']
        self.rel = model['rel_name']

    def detect_relation(self, subj, obj, query_type):
        """
        Get a list of relations sorted by confidence scores.
        Use underscore to indicate a space for multi-word terms.
        :param subj: subject of the triple.
        :param obj: object of the triple.
        :param query_type:
            max: outputs the max scoring of all relations
            sum: outputs the sum of scores of all relations (useful if you only need a general
                relatedness score for two terms)
            all: outputs scores for all relations, sorted by score
            topfive: outputs scores for the top five highest-scoring relations
            {relation_name}: for a specific relation name (e.g., "Causes"), outputs score for the tuple with that relation
        :return: results depending on the input query_type.
        """
        valid_query_types = ["max", "sum", "all", "topfive"]
        if query_type not in valid_query_types:
            logger.error("Unsupported query types. Should be one of " + str(valid_query_types))

        result = score(subj, obj, self.words, self.We, self.rel, self.Rel,
                       self.Weight, self.Offset, query_type, verbose=False)
        return result

    def score_triple(self, subj, obj, query_type):
        """
        Get a list of relations sorted by confidence scores.
        Use underscore to indicate a space for multi-word terms.
        :param subj: subject of the triple.
        :param obj: object of the triple.
        :param query_type:
            max: outputs the max scoring of all relations
            sum: outputs the sum of scores of all relations (useful if you only need a general
                relatedness score for two terms)
            all: outputs scores for all relations, sorted by score
            topfive: outputs scores for the top five highest-scoring relations
                {relation_name}: for a specific relation name (e.g., "Causes"), outputs score for the tuple with that relation
        :return: results depending on the input query_type.
        """
        valid_query_types = ["max", "sum", "all", "topfive"]
        if query_type not in valid_query_types:
            logger.error("Unsupported query types. Should be one of " + str(valid_query_types))

        result = score(subj, obj, self.words, self.We, self.rel, self.Rel,
                       self.Weight, self.Offset, query_type, verbose=False)
        return result


if __name__ == "__main__":
    scorer = CkbcTripleScorer()
    print(scorer.detect_relation(subj="a_bird", obj="flying_in_the_sky", query_type="max"))
    print(scorer.detect_relation(subj="a_bird", obj="flying_in_the_sky", query_type="sum"))
    print(scorer.detect_relation(subj="a_bird", obj="flying_in_the_sky", query_type="all"))
    print(scorer.detect_relation(subj="a_bird", obj="flying_in_the_sky", query_type="topfive"))

