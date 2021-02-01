#!/bin/bash

# Args:
# $1: input dir
# $2: output dir
# $3: number of lines per file

set -e

mkdir "$2"
for file in "$1"/*.txt; do
  echo $file
  fname="$(basename $file .txt)"
  split -l $3 --additional-suffix .txt -d "$1"/"${fname}.txt" "$2"/"${fname}_"
done
