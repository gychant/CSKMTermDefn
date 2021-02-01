"""
Extract ConceptNet subsets
Adapted from https://github.com/INK-USC/KagNet/blob/master/conceptnet/extract_cpnet.py
and https://github.com/INK-USC/KagNet/blob/master/conceptnet/merge_relation.txt
"""

import json
from tqdm import tqdm


relation_groups = [
    'atlocation/locatednear',
    'capableof',
    'causes/causesdesire/*motivatedbygoal',
    'createdby',
    'desires',
    'antonym/distinctfrom',
    'hascontext',
    'hasproperty',
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'isa/instanceof/definedas',
    'madeof',
    'notcapableof',
    'notdesires',
    'partof/*hasa',
    'relatedto/similarto/synonym',
    'usedfor',
    'receivesaction',
]


def load_merge_relation():
    relation_mapping = dict()
    for line in relation_groups:
        ls = line.strip().split('/')
        rel = ls[0]
        for l in ls:
            if l.startswith("*"):
                relation_mapping[l[1:]] = "*" + rel
            else:
                relation_mapping[l] = rel
    return relation_mapping


def get_entity_name(s):
    """
    Detect entity name from the raw entity string.
    :param s: Entity string
    :return: entity name
    """
    tokens = s.split("/")
    name = tokens[3]
    return name


def get_part_of_speech(s):
    """
    Detect part-of-speech encoding from an entity string, if present.
    :param s: Entity string
    :return: part-of-speech encoding
    """
    tokens = s.split("/")
    if len(tokens) <= 4:
        return ""

    pos_enc = tokens[4]
    if pos_enc == "n" or pos_enc == "a" or pos_enc == "v" or pos_enc == "r":
        return pos_enc
    return ""


def extract_english(conceptnet_path, output_jsonl_path, output_vocab_path=None,
                    conceptnet_core=False):
    """
    Extract subset of ConceptNet and build concept vocabulary.
    """
    relation_mapping = load_merge_relation()

    cpnet_vocab = dict()  # store word and the pos tags
    with open(conceptnet_path, 'r', encoding="utf8") as fin, \
            open(output_jsonl_path, 'w', encoding="utf8") as fout:
        for line in tqdm(fin):
            toks = line.strip().split('\t')
            head_str = toks[2]
            tail_str = toks[3]
            rel_str = toks[1]
            source = json.loads(toks[4])["dataset"]

            if head_str.startswith('/c/en/') and tail_str.startswith('/c/en/'):
                triple = dict()
                rel = rel_str.split("/")[-1].lower()
                head_pos = get_part_of_speech(head_str)
                tail_pos = get_part_of_speech(tail_str)
                head = get_entity_name(head_str).lower()
                tail = get_entity_name(tail_str).lower()

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue
                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue
                if rel not in relation_mapping:
                    continue

                # ignore triples not from conceptnet core
                if conceptnet_core and "conceptnet" not in source:
                    continue

                rel = relation_mapping[rel]
                if rel.startswith("*"):
                    head, tail, rel = tail, head, rel[1:]

                triple["head"] = head
                triple["tail"] = tail
                triple["rel"] = rel
                triple["head_pos"] = head_pos
                triple["tail_pos"] = tail_pos
                fout.write(json.dumps(triple) + '\n')

                for w, pos in zip([head, tail], [head_pos, tail_pos]):
                    if w not in cpnet_vocab:
                        cpnet_vocab[w] = set()
                    if pos != "":
                        cpnet_vocab[w].add(pos)
    print(f'Saved ConceptNet json file to {output_jsonl_path}')

    if output_vocab_path:
        with open(output_vocab_path, 'w') as f:
            for word in sorted(list(cpnet_vocab.keys())):
                f.write(word + '\t' + ",".join(sorted(list(cpnet_vocab[word]))) + "\n")
        print(f'Saved concept vocabulary to {output_vocab_path}')


if __name__ == "__main__":
    print("Extracting English subset of ConceptNet core ...")
    extract_english(conceptnet_path="./data/conceptnet/conceptnet-assertions-5.7.0.csv",
                    output_jsonl_path="./data/conceptnet/conceptnet_en_core.jsonl",
                    output_vocab_path="./data/conceptnet/conceptnet_vocab_pos_tag_core.tsv",
                    conceptnet_core=True)

    print("Extracting the full English subset of ConceptNet ...")
    extract_english(conceptnet_path="./data/conceptnet/conceptnet-assertions-5.7.0.csv",
                    output_jsonl_path="./data/conceptnet/conceptnet_en.jsonl",
                    conceptnet_core=False)
    print("DONE")

