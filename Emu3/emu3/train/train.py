# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import os
import os.path as osp
import pathlib
from typing import List, Optional, Tuple, Union

import numpy as np

import torch
from torch.nn import CrossEntropyLoss

import transformers as tf
from transformers.modeling_outputs import CausalLMOutputWithPast

from peft import LoraConfig, get_peft_model

import sys
sys.path.append("/lid/home/saydalie/multimodal_cot/Emu3")

from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM
from emu3.train.datasets import Emu3FeatureDataset

import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

BOV_ID, EOV_ID = 151854, 184621

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.05)
    apply_loss_on_only_vision: bool = field(default=True)
    apply_loss_on_only_text: bool = field(default=False)
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)


@dataclass
class TrainingArguments(tf.TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default="fa2")
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)


def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None else
        setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)

class Emu3ForCausalLMCustom(Emu3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.image_loss_log = []
        self.text_loss_log = []
    
    def reset_loss_log(self):
        self.image_loss_log = []
        self.text_loss_log = []
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        text_loss = 0
        image_loss = 0

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            # Pick image tokens: tokens that are in the range [bov, eov]
            image_token_mask = torch.logical_and(shift_labels >= BOV_ID, shift_labels <= EOV_ID)
            # Text tokens are everything else
            text_token_mask = ~image_token_mask

            # Compute separate losses
            loss_fct = CrossEntropyLoss()
            if text_token_mask.any():
                text_loss = loss_fct(shift_logits[text_token_mask], shift_labels[text_token_mask]).item()
                self.text_loss_log.append(text_loss)

            if image_token_mask.any():
                image_loss = loss_fct(shift_logits[image_token_mask], shift_labels[image_token_mask]).item()
                self.image_loss_log.append(image_loss)

            # Apply a weight of 0.5 to the loss associated with vision tokens
            # loss = text_loss + 0.5 * image_loss
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class TrainerCustom(tf.Trainer):
    def log(self, logs):
        # Retrieve the last computed losses from the model
        if self.model.image_loss_log and self.model.text_loss_log and ('loss' in logs):
            logs['image_loss'] = np.mean(self.model.image_loss_log)
            logs['text_loss'] = np.mean(self.model.text_loss_log)
            self.model.reset_loss_log()
        elif 'loss' not in logs:
            self.model.reset_loss_log()
        
        super().log(logs)
        
def train():
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_config = Emu3Config.from_pretrained(model_args.model_name_or_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate

    os.environ["WANDB_DIR"] = osp.join(training_args.output_dir, "wandb")
    os.environ["WANDB_RESUME"] = "allow"

    model = Emu3ForCausalLMCustom.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else None,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
    )

    # apply lora
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
    )

    # https://discuss.huggingface.co/t/peft-lora-gpt-neox-backward-pass-failing/35641
    for param in model.parameters():
        # freeze base model's layers
        param.requires_grad = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )

    train_dataset = Emu3FeatureDataset(data_args, tokenizer=tokenizer)
    print(f"bov: {train_dataset.bov}, eov: {train_dataset.eov}")

    trainer = TrainerCustom(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
