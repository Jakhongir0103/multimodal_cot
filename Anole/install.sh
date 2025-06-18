#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

pip install --upgrade setuptools pip packaging

cd transformers
pip install -e .
cd ..
pip install -r requirements.txt
pip install peft==0.12.0