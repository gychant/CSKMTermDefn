"""
Adapted from https://github.com/JoshFeldman95/Extracting-CK-from-Large-LM/blob/master/sentences.py
Collection of classes for use in generating sentences from relational triples
"""

from torch.utils.data import Dataset, DataLoader
from torch import nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, \
                                    GPT2LMHeadModel, GPT2Tokenizer
import torch
import csv
import re
import numpy as np
import requests
import json
from pathlib import Path
import spacy
# conda install -c bioconda mysqlclient -> to install mysql_config
# pip install pattern should work
from pattern.en import conjugate, PARTICIPLE, referenced, INDEFINITE, pluralize


raw_rel2text = {
    'AtLocation': 'at location',
    'CapableOf': 'capable of',
    'Causes': 'causes',
    'CausesDesire': 'causes desire',
    'CreatedBy': 'created by',
    'DefinedAs': 'defined as',
    'DesireOf': 'desire of',
    'Desires': 'desires',
    'HasA': 'has a',
    'HasFirstSubevent': 'has first subevent',
    'HasLastSubevent': 'has last subevent',
    'HasPainCharacter': 'has pain character',
    'HasPainIntensity': 'has pain intensity',
    'HasPrerequisite': 'has prerequisite',
    'HasProperty': 'has property',
    'HasSubevent': 'has subevent',
    'InheritsFrom': 'inherits from',
    'InstanceOf': 'instance of',
    'IsA': 'is a',
    'LocatedNear': 'located near',
    'LocationOfAction': 'location of action',
    'MadeOf': 'made of',
    'MotivatedByGoal': 'motivated by goal',
    'NotCapableOf': 'not capable of',
    'NotDesires': 'not desires',
    'NotHasA': 'not has a',
    'NotHasProperty': 'not has property',
    'NotIsA': 'not is a',
    'NotMadeOf': 'not made of',
    'PartOf': 'part of',
    'ReceivesAction': 'receives action',
    'RelatedTo': 'related to',
    'SymbolOf': 'symbol of',
    'UsedFor': 'used for'
}

rel2uppercase = {k.lower(): k for k, v in raw_rel2text.items()}


class CommonsenseTuples(Dataset):
    """ Base class for generating sentences from relational triples """

    def __init__(self, tuple_dir, device, language_model=None, template_loc=None,
                 relationwise=True, use_local_model=False):
        """
        Args:
            tuple_dir (string): Path to the csv file with commonsense tuples
        """
        self.sep_token = '[SEP]'
        self.start_token = '[CLS]'
        self.mask_token = '[MASK]'
        self.pad_token = '[PAD]'

        # Load pre-trained model tokenizer (vocabulary)
        if use_local_model:
            self.tokenizer = BertTokenizer.from_pretrained("../models/BertForMaskedLM")
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids([self.pad_token])[0]

        self.max_len = 20
        # self.stop_tokens = ['the', 'a', 'an']
        self.stop_tokens = ['the', 'an']
        self.device = device

        self.model = language_model
        if self.model is not None:
            self.model.eval()
            self.model.to(self.device)

        self.template_loc = template_loc
        self.tuples = []

        if relationwise:
            with open(tuple_dir) as f:
                for line in f:
                    rel, head, tail = line.strip().split("\t")
                    self.tuples.append((rel2uppercase[rel], head, tail))
        else:
            relations = [
                'AtLocation', 'CapableOf', 'Causes', 'CausesDesire',
                'CreatedBy', 'DefinedAs', 'Desires', 'HasA',
                'HasFirstSubevent', 'HasLastSubevent', 'HasPrerequisite', 'HasProperty',
                'HasSubevent', 'InstanceOf', 'IsA',
                'LocatedNear', 'MadeOf', 'MotivatedByGoal',
                'NotCapableOf', 'NotDesires', 'NotHasProperty',
                'NotIsA', 'NotMadeOf', 'PartOf', 'ReceivesAction',
                'SymbolOf', 'UsedFor']
            self.num_relations = len(relations)

            # Load tuples
            all_tuples = []
            with open(tuple_dir) as f:
                for line in f:
                    head, tail = line.strip().split("\t")
                    all_tuples.append((head, tail))
                    # for rel in relations:
                    #    self.tuples.append((rel, head, tail))

            # sample 1000 tuples
            import random
            random.seed(123)
            num_tuples = 1000
            random.shuffle(all_tuples)

            save_path = "../data/wiktionary_sampled_test_triples.txt"
            f_out = open(save_path, "w")
            for head, tail in all_tuples[:num_tuples]:
                for rel in relations:
                    self.tuples.append((rel, head, tail))
                    f_out.write("\t".join([head, rel, tail]))
                    f_out.write("\n")
            f_out.close()
            print("Saved sampled triples to {}".format(save_path))

        def collate_fn(batch):

            def _pad_sequences(sequences, pad_token_id):
                """
                Pad to the longest sequence in the batch
                """
                max_len = max([len(seq) for seq in sequences])
                for idx in range(len(sequences)):
                    while len(sequences[idx]) < max_len:
                        sequences[idx].append(pad_token_id)

            batch_indexed_sent, batch_indexed_masked_list, \
                batch_segments_ids, batch_head_masked_ids, batch_tail_masked_ids = zip(*batch)
            batch_masked_head, batch_masked_tail, batch_masked_both = zip(*batch_indexed_masked_list)

            _pad_sequences(batch_indexed_sent, self.pad_token_id)
            _pad_sequences(batch_masked_head, self.pad_token_id)
            _pad_sequences(batch_masked_tail, self.pad_token_id)
            _pad_sequences(batch_masked_both, self.pad_token_id)
            _pad_sequences(batch_segments_ids, pad_token_id=1)
            return (
                torch.tensor(batch_indexed_sent, device=self.device),
                (
                    torch.tensor(batch_masked_head, device=self.device),
                    torch.tensor(batch_masked_tail, device=self.device),
                    torch.tensor(batch_masked_both, device=self.device)
                ),
                torch.tensor(batch_segments_ids, device=self.device),
                batch_head_masked_ids,
                batch_tail_masked_ids
            )
        self.collate_fn = collate_fn

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        # get tuple
        r, t1, t2 = self.tuples[idx][:3]

        # apply template
        # try:
        sent, t1, t2 = self.apply_template(r, t1, t2, use_best_candidate=True)
        # except (json.JSONDecodeError) as e:
        #    return (-1,-1,-1,-1)
        # apply start and end tokens
        sent = f"{self.start_token} {sent}. {self.sep_token}"

        # tokenize sentences and t1 and t2
        tokenized_sent = self.tokenizer.tokenize(sent)
        tokenized_t1 = self.tokenizer.tokenize(t1)
        tokenized_t2 = self.tokenizer.tokenize(t2)

        # mask sentence
        masked_head, masked_tail, masked_both = \
            self.mask_sentence(tokenized_sent, tokenized_t1, tokenized_t2)

        # get segment ids
        segments_ids = self.get_segment_ids(tokenized_sent)

        # convert tokens to ids
        indexed_sent = self.tokenizer.convert_tokens_to_ids(tokenized_sent)

        # (masked_head, masked_tail, masked_both)
        indexed_masked_head = self.tokenizer.convert_tokens_to_ids(masked_head)
        indexed_masked_tail = self.tokenizer.convert_tokens_to_ids(masked_tail)
        indexed_masked_both = self.tokenizer.convert_tokens_to_ids(masked_both)

        indexed_masked_list = (indexed_masked_head, indexed_masked_tail, indexed_masked_both)
        head_masked_ids = [idx for idx, token in enumerate(indexed_masked_head) if token == 103]
        tail_masked_ids = [idx for idx, token in enumerate(indexed_masked_tail) if token == 103]

        return indexed_sent, indexed_masked_list, segments_ids, head_masked_ids, tail_masked_ids

    def mask(self, tokenized_sent, tokenized_to_mask):
        tokenized_masked = tokenized_sent.copy()
        for idx_sent in range(len(tokenized_masked)-len(tokenized_to_mask)):
            match = []
            for idx_mask in range(len(tokenized_to_mask)):
                match.append(tokenized_masked[idx_sent+idx_mask] == tokenized_to_mask[idx_mask])
            if all(match):
                for idx_mask in range(len(tokenized_to_mask)):
                    if tokenized_masked[idx_sent+idx_mask] not in self.stop_tokens:
                        tokenized_masked[idx_sent+idx_mask] = self.mask_token
        return tokenized_masked

    def mask_sentence(self, tokenized_sent, tokenized_t1, tokenized_t2):
        masked_sent_list = []
        # mask head
        masked_sent_list.append(self.mask(tokenized_sent, tokenized_t2))
        # mask tail
        masked_sent_list.append(self.mask(tokenized_sent, tokenized_t1))
        # mask both
        tokenized_sent = self.mask(tokenized_sent, tokenized_t1)
        masked_sent_list.append(self.mask(tokenized_sent, tokenized_t2))
        return masked_sent_list

    def get_segment_ids(self, tokenized_sent):
        segments_ids = []
        segment = 0
        for word in tokenized_sent:
            segments_ids.append(segment)
            if word == self.sep_token:
                segment += 1
        return segments_ids

    def id_to_text(self, sent):
        if type(sent) == torch.Tensor:
            tokens = [self.tokenizer.ids_to_tokens[sent[idx].item()] for idx in range(len(sent))]
        else:
            tokens = [self.tokenizer.ids_to_tokens[sent[idx]] for idx in range(len(sent))]
        return " ".join(tokens)

    def apply_template(self, relation, head, tail):
        """ To be overriden, returning the sentence, head, and tail """
        return NotImplementedError


class DirectTemplate(CommonsenseTuples):
    """ Sentence generation approach via direct concatenation. I.e., head and
    tail are concatenated to the ends of the relation as strings """

    def __init__(self, *args, template_loc=None, language_model=None):
        super().__init__(*args, template_loc=template_loc, language_model=language_model)
        self.regex = '[A-Z][^A-Z]*'

    def apply_template(self, relation, head, tail):
        template = " ".join(re.findall(self.regex, relation))
        return ' '.join([head, template, tail]), head, tail


class PredefinedTemplate(CommonsenseTuples):
    """ Sentence generation via predefined template for each relation """

    def __init__(self, *args, template_loc='relation_map.json', grammar=False, language_model=None):
        super().__init__(*args)
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.grammar = grammar
        with open(template_loc, 'r') as f:
            self.templates = json.load(f)

    def clean_text(self, words):
        new_words = words.split(' ')
        doc = self.nlp(words)
        first_word_POS = doc[0].pos_
        if first_word_POS == 'VERB':
            new_words[0] = conjugate(new_words[0], tense=PARTICIPLE)
        if first_word_POS == 'NOUN' or first_word_POS == 'ADJ':
            if new_words[0] != 'a' or new_words[0] != 'an':
                new_words[0] = referenced(new_words[0])
        elif first_word_POS == 'NUM' and len(new_words) > 1:
            new_words[1] = pluralize(new_words[1])
        return ' '.join(new_words)

    def apply_template(self, relation, head, tail):
        if self.grammar:
            head = self.clean_text(head)
            tail = self.clean_text(tail)
        sent = self.templates[relation].format(head, tail)
        return sent, head, tail


class EnumeratedTemplate(CommonsenseTuples):
    """ Sentence generation with coherency ranking """

    def __init__(self, *args, language_model=None, template_loc='./relation_map_multiple.json',
                 use_local_model=False):
        super().__init__(*args, language_model=language_model, template_loc=template_loc,
                         use_local_model=use_local_model)
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        if use_local_model:
            self.enc = GPT2Tokenizer.from_pretrained("../models/GPT2LMHeadModel")
        else:
            self.enc = GPT2Tokenizer.from_pretrained('gpt2')

        with open(self.template_loc, 'r') as f:
            self.templates = json.load(f)

    def apply_template(self, relation, head, tail, use_best_candidate=True):
        if use_best_candidate:
            candidate_sents = self.get_candidates(relation, head, tail)
            sent, head, tail = self.get_best_candidate(candidate_sents)
        else:
            sent = "{} {} {}".format(head, raw_rel2text[relation], tail)
        return sent, head, tail

    def get_candidates(self, relation, head, tail):
        heads = self.formats(head)
        tails = self.formats(tail)
        templates = self.templates[relation]
        candidate_sents = []

        for h in heads:
            for t in tails:
                for temp in templates:
                    candidate_sents.append((temp.format(h, t), h, t))

        return candidate_sents

    def formats(self, phrase):
        doc = self.nlp(phrase)
        first_word_POS = doc[0].pos_

        tokens = phrase.split(' ')
        new_tokens = tokens.copy()

        new_phrases = []
        # original
        new_phrases.append(' '.join(new_tokens))

        # with indefinite article
        if first_word_POS == 'NOUN' or first_word_POS == 'ADJ':
            new_tokens[0] = referenced(tokens[0])
            new_phrases.append(' '.join(new_tokens))
        # with definite article
        if first_word_POS == 'NOUN' or first_word_POS == 'ADJ':
            new_tokens[0] = "the "+tokens[0]
            new_phrases.append(' '.join(new_tokens))
        # as gerund
        if first_word_POS == 'VERB':
            new_tokens[0] = conjugate(tokens[0], tense=PARTICIPLE)
            new_phrases.append(' '.join(new_tokens))
            if len(tokens) > 1:
                if tokens[1] == 'to' and len(tokens) > 2:
                    new_tokens[2] = referenced(tokens[2])
                else:
                    new_tokens[1] = referenced(tokens[1])
            new_phrases.append(' '.join(new_tokens))
            new_tokens[0] = tokens[0]
            new_phrases.append(' '.join(new_tokens))

        # account for numbers
        if first_word_POS == 'NUM' and len(tokens) > 1:
            new_tokens[1] = pluralize(tokens[1])
            new_phrases.append(' '.join(new_tokens))
        return new_phrases

    def get_best_candidate(self, candidate_sents):
        candidate_sents.sort(key=self.score_sent, reverse=True)
        return candidate_sents[0]

    def score_sent(self, candidate):
        sent, _, _ = candidate
        sent = ". "+sent

        try:
            tokens = self.enc.encode(sent)
        except KeyError:
            return 0

        context = torch.tensor(tokens, dtype=torch.long, device=self.device).reshape(1,-1)
        logits, _ = self.model(context)
        log_probs = logits.log_softmax(2)
        sentence_log_prob = 0

        for idx, c in enumerate(tokens):
            if idx > 0:
                sentence_log_prob += log_probs[0, idx-1, c]

        return sentence_log_prob.item() / (len(tokens)**0.2)

