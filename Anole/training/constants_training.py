import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Raw dataset (specify the path of your raw dataset)
DATASET_RAW_DIR = Path(f"/lid/home/saydalie/multimodal_cot/LLM-PuzzleTest/PuzzleVQA/data/")

# Tokenized dataset (specify the path that you want to store your tokenized dataset)
DATASET_TOKENIZED_DIR = Path("/lid/home/saydalie/multimodal_cot/anole/data/")

# Anole HF path (specify the path that you want to store your fine-tuned Anole hugging face checkpoint)
RESULTS_DIR = Path("/lid/home/saydalie/multimodal_cot/results/anole-hf_trained/")

# Anole torch path (specify the path that you want to store your Anole torch checkpoint)
ANOLE_PATH_TORCH = Path("/lid/home/saydalie/multimodal_cot/Anole-7b-v0.1/")

# Anole HF path (specify the path that you want to store your Anole hugging face checkpoint)
ANOLE_PATH_HF = Path("/lid/home/saydalie/multimodal_cot/Anole-7b-v0.1-hf/")

# Anole HF path (specify the path that you want to store your fine-tuned Anole hugging face checkpoint)
ANOLE_PATH_HF_TRAINED = Path("/lid/home/saydalie/multimodal_cot/models/anole-hf_trained/")
