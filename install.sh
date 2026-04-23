#!/bin/bash

sudo rm -r venv
python3 -m venv venv
source venv/bin/activate

python3 -m pip install --upgrade pip

pip install torch tqdm psutil lion-pytorch numpy transformers sentencepiece
