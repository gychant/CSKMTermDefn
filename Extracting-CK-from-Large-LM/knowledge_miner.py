"""
Adopted from https://github.com/JoshFeldman95/Extracting-CK-from-Large-LM/blob/master/knowledge_miner.py
"""

from copy import deepcopy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from torch.utils.data import Dataset, DataLoader


class KnowledgeMiner:

    def __init__(self, dev_data_path, device, Template, bert, **kwarg):
        """ Creates a class instance for doing KBC with a given template and
        HuggingFace bert model. Template classes defined in `sentences.py` """
        self.sentences = Template(
            dev_data_path,
            device,
            **kwarg
        )

        # self.batch_size = self.sentences.num_relations
        self.batch_size = 64
        self.data_loader = DataLoader(self.sentences, batch_size=self.batch_size,
                                      collate_fn=self.sentences.collate_fn)

        bert = torch.nn.DataParallel(bert)
        bert.eval()
        bert.to(device)
        self.bert = bert
        self.device = device

    def make_predictions(self, return_dataframe=False):
        data = []
        for idx, batch in tqdm(enumerate(self.data_loader)):
            (sent, (masked_head, masked_tail, masked_both), ids, head_masked_ids, tail_masked_ids) = batch

            # conditional
            logprob_tail_conditional = self.batch_predict(sent, masked_tail, ids, tail_masked_ids)
            logprob_head_conditional = self.batch_predict(sent, masked_head, ids, head_masked_ids)
            # marginal
            logprob_tail_marginal = self.batch_predict(sent, masked_both, ids, tail_masked_ids)
            logprob_head_marginal = self.batch_predict(sent, masked_both, ids, head_masked_ids)

            for b in range(sent.size(0)):
                # NLL = - logprob_tail_conditional[b] / len(tail_masked_ids[b])

                # average approximations of PMI(t,h|r) and PMI(h,t|r)
                mutual_inf = logprob_tail_conditional[b] - logprob_tail_marginal[b]
                mutual_inf += logprob_head_conditional[b] - logprob_head_marginal[b]
                mutual_inf /= 2.

                data.append(mutual_inf)

        if return_dataframe:
            df = pd.DataFrame(data, columns=(
                'nll', 'tail_conditional', 'tail_marginal',
                'head_conditional', 'head_marginal', 'mut_inf', 'sent'))
            return df
        return data

    def predict(self, sent, masked, ids, masked_ids):
        logprob = 0
        masked = deepcopy(masked)
        masked_ids = masked_ids.copy()

        for _ in range(len(masked_ids)):
            # make prediction
            pred = self.bert(masked.reshape(1, -1), ids.reshape(1, -1)).log_softmax(2)

            # get log probs for each token
            max_log_prob = -np.inf

            for idx in masked_ids:
                if pred[0, idx, sent[idx]] > max_log_prob:
                    most_likely_idx = idx
                    max_log_prob = pred[0, idx, sent[idx]]

            logprob += max_log_prob
            masked[most_likely_idx] = sent[most_likely_idx]
            masked_ids.remove(most_likely_idx)

        return logprob

    def batch_predict(self, sent, masked, ids, masked_ids):
        masked = deepcopy(masked)
        masked_ids = deepcopy(list(masked_ids))
        max_num_masked_ids = max([len(ids) for ids in masked_ids])
        logprobs = [0] * sent.size(0)

        for _ in range(max_num_masked_ids):
            with torch.no_grad():
                # make prediction
                pred = self.bert(input_ids=masked,
                                 attention_mask=1 - ids,
                                 token_type_ids=ids).log_softmax(2)
                pred = pred.detach()

            max_log_probs = [-np.inf] * sent.size(0)

            for i in range(len(masked_ids)):
                most_likely_idx = -1
                for idx in masked_ids[i]:
                    if pred[i, idx, sent[i][idx]] > max_log_probs[i]:
                        most_likely_idx = idx
                        max_log_probs[i] = pred[i, idx, sent[i][idx]]

                if most_likely_idx in masked_ids[i]:
                    logprobs[i] += max_log_probs[i].item()
                    masked[i][most_likely_idx] = sent[i][most_likely_idx]
                    masked_ids[i].remove(most_likely_idx)
        return logprobs

