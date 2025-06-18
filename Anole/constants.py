import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

ckpt_path = Path(os.getenv("CKPT_PATH", "/lid/home/saydalie/multimodal_cot/Anole-7b-v0.1/"))

MODEL_7B_PATH = ckpt_path / "models" / "7b"

MODEL_30B_PATH = ckpt_path / "models" / "30b"

TOKENIZER_TEXT_PATH = ckpt_path / "tokenizer" / "text_tokenizer.json"

TOKENIZER_IMAGE_PATH = ckpt_path / "tokenizer" / "vqgan.ckpt"

TOKENIZER_IMAGE_CFG_PATH = ckpt_path / "tokenizer" / "vqgan.yaml"
