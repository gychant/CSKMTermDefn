"""
Adopted from https://github.com/JoshFeldman95/Extracting-CK-from-Large-LM/blob/master/wikipedia_experiments.py
"""

import os
import sys
import torch

from sentences import DirectTemplate, PredefinedTemplate, EnumeratedTemplate
from knowledge_miner import KnowledgeMiner
from pytorch_pretrained_bert import BertForMaskedLM, GPT2LMHeadModel

bert_model = 'bert-large-uncased'
gpt2_model = 'gpt2'

template_repo = os.path.join(os.path.dirname(__file__), 'templates')
single_templates = 'relation_map.json'
multiple_templates = 'relation_map_multiple.json'

data_repo = './data/wiktionary_relationwise_candidates_by_pos_tag_core'
# candidate_file = 'wiktionary_candidates_by_pos_tag.txt'


def run_experiment(template_type, knowledge_miners, save_csv=False):
    print(f'make predictions using {template_type} templates...')
    ck_miner = knowledge_miners[template_type]

    if save_csv:
        df = ck_miner.make_predictions(return_dataframe=True)
        df.to_csv(os.path.join(data_repo, "{}.csv".format(template_type)))
    else:
        predictions = ck_miner.make_predictions(return_dataframe=False)
        return predictions


def mine_triples(device, input_file, output_file, use_local_model=False):
    if use_local_model:
        print('loading BERT...')
        bert = BertForMaskedLM.from_pretrained("../models/BertForMaskedLM")
        print('loading GPT2...')
        gpt = GPT2LMHeadModel.from_pretrained("../models/GPT2LMHeadModel")
    else:
        print('loading BERT...')
        bert = BertForMaskedLM.from_pretrained(bert_model)
        print('loading GPT2...')
        gpt = GPT2LMHeadModel.from_pretrained(gpt2_model)

    """
        'concat': KnowledgeMiner(
            os.path.join(data_repo, candidate_file),
            device,
            DirectTemplate,
            bert
        ),
        'template': KnowledgeMiner(
            os.path.join(data_repo, candidate_file),
            device,
            PredefinedTemplate,
            bert,
            grammar=False,
            template_loc=os.path.join(template_repo, single_templates)
        ),
        'template_grammar': KnowledgeMiner(
            os.path.join(data_repo, candidate_file),
            device,
            PredefinedTemplate,
            bert,
            grammar=True,
            template_loc=os.path.join(template_repo, single_templates)
        ),
    """

    knowledge_miners = {
        'coherency': KnowledgeMiner(
            input_file,
            device,
            EnumeratedTemplate,
            bert,
            language_model=gpt,
            template_loc=os.path.join(template_repo, multiple_templates),
            use_local_model=use_local_model
        )
    }

    for template_type in knowledge_miners.keys():
        predictions = run_experiment(template_type, knowledge_miners)
        triples = knowledge_miners[template_type].sentences.tuples
        scored_samples = list(zip(triples, predictions))
        scored_samples.sort(key=lambda x: x[1], reverse=True)
        with open(output_file, "w") as f:
            for triple, pred in scored_samples:
                rel, head, tail = triple
                triple = (rel.lower(), head, tail)
                f.write("\t".join(triple) + "\t" + "{:.5f}".format(pred))
                f.write("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file_path', type=str, help='test_file_path')
    parser.add_argument('--test_file_name', type=str, help='test_file_name')
    parser.add_argument('--output_dir', type=str, help='output_dir')
    args = parser.parse_args()

    test_file_path = args.test_file_path
    test_file_name = args.test_file_name
    output_dir = args.output_dir
    print("Evaluating file {} ...".format(test_file_name))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(output_dir):
        # concurrent operation
        try:
            os.makedirs(output_dir)
        except:
            pass

    mine_triples(device, input_file=test_file_path,
                 output_file=os.path.join(output_dir, test_file_name),
                 use_local_model=False)

