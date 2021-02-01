# Commonsense Knowledge Mining from Term Definitions
This repository contains the source code released along with our paper [Commonsense Knowledge Mining from Term Definitions](https://usc-isi-i2.github.io/AAAI21workshop/papers/Liang_CSKGsAAAI-21.pdf) at the Commonsense Knowledge Graphs (CSKGs) Workshop of AAAI 2021 (https://usc-isi-i2.github.io/AAAI21workshop/). 

If you find our paper and code useful for your research, please cite
```
@inproceedings{cskmtermdefn-cskgaaai21,
  title     = {Commonsense Knowledge Mining from Term Definitions},
  author    = {Zhicheng Liang and Deborah L. McGuinness},
  booktitle = {The Commonsense Knowledge Graphs (CSKGs) Workshop of AAAI 2021},
  year      = {2021}
}
```

## Usage
### Preparation
#### 1. Create a python3 virtual environment and install the dependencies
```
pip install -r requirements.txt
```
and set python path to the repo root directory using
```
export PYTHONPATH=.
```

#### 2. Download resources
```
sh scripts/download_ckbc_data.sh data/CKBC
sh scripts/download_conceptnet.sh data/conceptnet
```

#### 3. Extract data from ConceptNet
```
python wiktionary/conceptnet.py
```

#### 4. Crawl Wiktionary term definitions
```
python wiktionary/crawl_term.py
```
This step is time consuming due to the large vocabulary.

### Mine Commonsense Knowledge Triples
#### 1. Analyze and collect POS tag sequence patterns from triples
```
python wiktionary/triple_pos_tag_analyzer.py
```

#### 2. Extract commonsense knowledge triples from term definitions using frequent POS tag patterns
```
python wiktionary/wiktionary_triple_extractor.py
```

### Score extracted candidate triples using different models
#### 1. Bilinear AVG model
```
python wiktionary/wiktionary_triple_evaluation.py
```

#### 2. KG-BERT model
The code is adapted from the [repo](https://github.com/yao8839836/kg-bert) of [KG-BERT: BERT for Knowledge Graph Completion](https://arxiv.org/abs/1909.03193). We add CKBC data for training. To train on CKBC data, run
```
sh kg-bert/train_ckbc.sh
```
After training, to evaluate on Wiktionary candidate triples, run
```
sh kg-bert/test_wiktionary.sh
```

#### 3. PMI model 
The code is adapted from the [repo](https://github.com/JoshFeldman95/Extracting-CK-from-Large-LM) of [Commonsense Knowledge Mining from Pretrained Models](https://arxiv.org/abs/1909.00505). 
We modify their implementation to support batch inference for efficiency. Given the large amount of candidate triples, we also leverage the cluster to run predictions in parallel with each node running on a smaller split input. 

To split candidate triple files, run
```
sh Extracting-CK-from-Large-LM/split_files.sh [input dir] [output dir] [number of lines per file]
```
To run prediction:
```
python Extracting-CK-from-Large-LM/wiktionary_experiment.py 
 --test_file_path [test_file_path] 
 --test_file_name [test_file_name] 
 --output_dir [output_dir] 
```
For example, to score triples of the AtLocation relation, run
```
python Extracting-CK-from-Large-LM/wiktionary_experiment.py 
 --test_file_path ./data/wiktionary_relationwise_candidates_by_pos_tag_core/atlocation.txt 
 --test_file_name atlocation.txt 
 --output_dir ./data/pmi_coherency_wiktionary_core 
```
If needed to merge scored split triple files back, run
```
python Extracting-CK-from-Large-LM/merge_split_files.py 
 --input_dir [input_dir] 
 --output_dir [output_dir] 
```
For example:
```
python Extracting-CK-from-Large-LM/merge_split_files.py 
 --input_dir ./data/pmi_coherency_wiktionary_core_split 
 --output_dir ./data/pmi_coherency_wiktionary_core 
```

### Evaluation
#### 1. Plot score distributions of different models
```
python wiktionary/score_dist_plot.py
```

#### 2. Compute the Kendall’s tau coefficients between pairs of models
```
python wiktionary/kendall_tau.py
```

#### 3. Check novelty of extracted triples
```
python wiktionary/check_novelty.py --model BILINEAR-AVG
python wiktionary/check_novelty.py --model KG-BERT
python wiktionary/check_novelty.py --model PMI
```

#### 4. Sample qualified candidate triples for manual evaluation
```
python wiktionary/sample_triple_subset.py
```

#### 5. Check novelty of triples sampled for manual evaluation
```
python wiktionary/check_novelty.py --model BILINEAR-AVG --samples_only
python wiktionary/check_novelty.py --model KG-BERT --samples_only
python wiktionary/check_novelty.py --model PMI --samples_only
```

### Contact
If you have any question regarding the code, feel free to create a github issue or email us (emails provided in the paper).


