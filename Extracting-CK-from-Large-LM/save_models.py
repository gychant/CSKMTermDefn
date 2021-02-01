
import os
import torch

from pytorch_pretrained_bert import BertTokenizer, GPT2Tokenizer
from pytorch_pretrained_bert import BertForMaskedLM, GPT2LMHeadModel


def save_model(model_to_save, tokenizer, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output_model_file = os.path.join(save_dir, "pytorch_model.bin")
    output_config_file = os.path.join(save_dir, "config.json")

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(save_dir)


bert_model = 'bert-large-uncased'
gpt2_model = 'gpt2'

print('loading BERT...')
bert = BertForMaskedLM.from_pretrained(bert_model)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print('loading GPT2...')
gpt = GPT2LMHeadModel.from_pretrained(gpt2_model)
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# save models
save_model(bert, bert_tokenizer, "../models/BertForMaskedLM")
save_model(gpt, gpt_tokenizer, "../models/GPT2LMHeadModel")
print("DONE")

