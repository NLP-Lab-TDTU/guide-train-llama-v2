import math
from dataclasses import dataclass, field
from typing import Optional
import os
import torch

from datasets import load_dataset
from datasets import load_from_disk
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import TrainerCallback
from transformers import HfArgumentParser


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    dataset = load_from_disk(data_args.dataset_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, verbose=False)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.pad_token = '<pad>'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        device_map={"": Accelerator().process_index},
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True
    )
    model.resize_token_embeddings(len(tokenizer))
    
    class SaveCheckpointCallback(TrainerCallback):
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            
        def on_epoch_end(self, args, state, control, **kwargs):
            if state.is_local_process_zero:
                checkpoint_name = "checkpoint-epoch-" + str(math.round(state.epoch))
                checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
    
    training_args.prediction_loss_only = True
    training_args.save_strategy = "steps"
    
    response_template = "<|im_start|>assistant"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, pad_to_multiple_of=8)
    
    trainer = SFTTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=128,
        data_collator=collator,
    )

    trainer.add_callback(SaveCheckpointCallback(model, tokenizer))
    
    with torch.autocast("cuda"): 
        trainer.train()

if __name__ == "__main__":
    main()
