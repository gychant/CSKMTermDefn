#!/bin/bash

# Download ConceptNet 5.7.0
# sh scripts/download_conceptnet.sh data/conceptnet

set -e
save_dir=$1

wget -P ${save_dir} https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz

echo "Unzip ..."
gunzip ${save_dir}/conceptnet-assertions-5.7.0.csv.gz
echo "Saved to ${save_dir}/conceptnet-assertions-5.7.0.csv.gz"

echo "Download & unzip DONE."
