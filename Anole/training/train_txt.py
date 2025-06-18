import os
import json
import torch
import deepspeed
import jsonlines

import sys
sys.path.append("/lid/home/saydalie/multimodal_cot")
sys.path.append("/lid/home/saydalie/multimodal_cot/anole")

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import ChameleonForCausalLM, Trainer, TrainingArguments

from training.constants_training import (
    ANOLE_PATH_HF,
    ANOLE_PATH_HF_TRAINED,
    DATASET_TOKENIZED_DIR
)
from peft import LoraConfig, get_peft_model

def exists_checkpoint(output_dir: str):
    if output_dir.exists() and os.listdir(output_dir):
        for folder in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir, folder)) and folder.startswith("checkpoint"):
                return True
    return False

# Define the dataset class
class TokenizedDataset(Dataset):
    def __init__(self, data_dir, pattern_names, start_idx=0, end_idx=9999):
        self.tokenized_data = []

        for pattern_name in pattern_names:
            data_file_path = os.path.join(data_dir, f"train/{pattern_name}_txt.jsonl")

            with jsonlines.open(data_file_path) as reader:
                for idx, obj in enumerate(reader):
                    if (idx < start_idx) or (idx > end_idx):
                        continue

                    self.tokenized_data.append(torch.tensor(obj['input_text_ids'], dtype=torch.long))
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]

# Define custom collate function
def collate_fn(batch):
    # pad_id = 1
    # left padding
    batch_flipped = [item.flip(0) for item in batch]
    batch_inputs_padded = pad_sequence(batch_flipped, batch_first=True, padding_value=1).flip(1)
    
    # Create attention masks
    attention_masks = torch.zeros_like(batch_inputs_padded, dtype=torch.long)
    attention_masks = attention_masks.masked_fill(batch_inputs_padded != 1, 1)
   
    return {'input_ids': batch_inputs_padded, 'attention_mask': attention_masks, 'labels': batch_inputs_padded.clone()}

# Initialize the datasets
pattern_names_train = ['color_number_hexagon', 'grid_number_color', 'rectangle_height_color', 'polygon_sides_number', 'triangle']
pattern_names_ood = ['color_grid', 'color_overlap_squares', 'rectangle_height_number', 'polygon_sides_color', 'shape_reflect']
train_dataset = TokenizedDataset(
    data_dir=DATASET_TOKENIZED_DIR, 
    pattern_names=pattern_names_train, 
    start_idx=0, 
    end_idx=4999
)
eval_dataset = {
    "in_domain": TokenizedDataset(
        data_dir=DATASET_TOKENIZED_DIR,
        pattern_names=pattern_names_train, 
        start_idx=5000, 
        end_idx=5099
    ), 
    "out_of_domain": TokenizedDataset(
        data_dir=DATASET_TOKENIZED_DIR,
        pattern_names=pattern_names_ood, 
        start_idx=0, 
        end_idx=99
    )
}
print(f"Datasets sizes: {len(train_dataset)}, {len(eval_dataset['in_domain'])}, {len(eval_dataset['out_of_domain'])}")

# `ChameleonForCausalLM` with custom loss function
class ChameleonForCausalLMCustom(ChameleonForCausalLM):
    def compute_loss(
            self, 
            logits: torch.LongTensor,
            labels: torch.LongTensor
        ):
        # # Disallow image tokens which does not include special begin-image and end-image tokens: Useful when training on text-only data
        # # Why use this: https://chatgpt.com/share/67c6af02-6884-8012-b688-2a85e09b5488
        # image_tokens = self.model.vocabulary_mapping.image_tokens
        # logits[:, :, image_tokens] = torch.finfo(logits.dtype).min

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return loss
    
# Load the model with Lora
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

model = ChameleonForCausalLMCustom.from_pretrained(
    ANOLE_PATH_HF,
    torch_dtype=torch.bfloat16
)

# model.config.max_position_embeddings = 6144
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Define training arguments
ds_config_path = os.path.join(os.path.dirname(__file__), "ds_config.json")
with open(ds_config_path, "r") as f:
    ds_config = json.load(f)

run_name = 'Anole-7B-5DS-CE-txt'
output_dir = ANOLE_PATH_HF_TRAINED / run_name

training_args = TrainingArguments(
    output_dir=str(output_dir),
    learning_rate=2e-5,
    num_train_epochs=20,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=4,
    save_strategy="steps",
    eval_strategy="steps",
    logging_strategy="steps",
    save_steps=0.05,
    eval_steps=0.05,
    logging_steps=0.01,
    eval_on_start=True,
    save_total_limit=3,
    bf16=True,
    fp16=False,
    push_to_hub=False,
    report_to='wandb', # wandb
    run_name=run_name
)
training_args.deepspeed=ds_config

# Initialize the Trainer with custom collate_fn
os.environ["WANDB_RUN_ID"] = run_name
os.environ["WANDB_RESUME"] = "allow"

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     # Shift so that tokens < n predict n
#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_labels = labels[..., 1:].contiguous()
#     # Flatten the tokens
#     loss_fct = CrossEntropyLoss()
#     shift_logits = shift_logits.view(-1, model.config.vocab_size)
#     shift_labels = shift_labels.view(-1)
#     # Enable model parallelism
#     shift_labels = shift_labels.to(shift_logits.device)
#     loss = loss_fct(shift_logits, shift_labels)
#     return {"eval/loss": loss.item()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    # compute_metrics=compute_metrics
)

# Train the model
if exists_checkpoint(output_dir):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# Save the model
model.save_pretrained(str(output_dir / 'final'))
