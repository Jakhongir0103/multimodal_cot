#!/bin/bash
export BASE_PATH="/lid/home/saydalie"

python -u $BASE_PATH/multimodal_cot/anole/training/data_tokenization.py > $BASE_PATH/_runai_out/data_tokenization.txt 2>&1