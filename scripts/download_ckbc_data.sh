#!/bin/bash

# Download CKBC resources from https://ttic.uchicago.edu/~kgimpel/commonsense.html

set -e
save_dir=$1

# training set of 100,000 tuples: train100k.txt.gz
wget -P ${save_dir} https://ttic.uchicago.edu/~kgimpel/comsense_resources/train100k.txt.gz
gunzip ${save_dir}/train100k.txt.gz

# training set of 300,000 tuples: train300k.txt.gz
wget -P ${save_dir} https://ttic.uchicago.edu/~kgimpel/comsense_resources/train300k.txt.gz
gunzip ${save_dir}/train300k.txt.gz

# training set of 600,000 tuples (conjugated and unconjugated forms): train600k.txt.gz
wget -P ${save_dir} https://ttic.uchicago.edu/~kgimpel/comsense_resources/train600k.txt.gz
gunzip ${save_dir}/train600k.txt.gz

# dev1 development set: dev1.txt.gz
wget -P ${save_dir} https://ttic.uchicago.edu/~kgimpel/comsense_resources/dev1.txt.gz
gunzip ${save_dir}/dev1.txt.gz

# dev2 development set: dev2.txt.gz
wget -P ${save_dir} https://ttic.uchicago.edu/~kgimpel/comsense_resources/dev2.txt.gz
gunzip ${save_dir}/dev2.txt.gz

# test set: test.txt.gz
wget -P ${save_dir} https://ttic.uchicago.edu/~kgimpel/comsense_resources/test.txt.gz
gunzip ${save_dir}/test.txt.gz

# Demo for scoring arbitrary tuples (see examples below): ckbc-demo.tar.gz
wget -P ${save_dir} https://ttic.uchicago.edu/~kgimpel/comsense_resources/ckbc-demo.tar.gz
tar xvzf ${save_dir}/ckbc-demo.tar.gz -C ${save_dir}
rm ${save_dir}/ckbc-demo.tar.gz
mv ${save_dir}/ckbc-demo/Bilinear*.pickle ${save_dir}/ckbc-demo/Bilinear.pickle

# Automatically-generated ConceptNet/Wikipedia tuples scored with our model (Bilinear AVG, trained on 600k training set):
# tuples created by automatically modifying ConceptNet training tuples: tuples.cn.txt.gz
wget -P ${save_dir} https://ttic.uchicago.edu/~kgimpel/comsense_resources/tuples.cn.txt.gz
gunzip ${save_dir}/tuples.cn.txt.gz

# tuples automatically extracted from Wikipedia sentences: tuples.wiki.tar.gz
wget -P ${save_dir} https://ttic.uchicago.edu/~kgimpel/comsense_resources/tuples.wiki.tar.gz
tar xvzf ${save_dir}/tuples.wiki.tar.gz -C ${save_dir}
rm ${save_dir}/tuples.wiki.tar.gz

# ConceptNet-trained word embeddings:
# actual word embeddings: embeddings.txt.gz
wget -P ${save_dir} https://ttic.uchicago.edu/~kgimpel/comsense_resources/embeddings.txt.gz
gunzip ${save_dir}/embeddings.txt.gz

# data for training word embeddings: embedding_training.txt.gz
wget -P ${save_dir} https://ttic.uchicago.edu/~kgimpel/comsense_resources/embedding_training.txt.gz
gunzip ${save_dir}/embedding_training.txt.gz

echo "Download & unzip DONE."
