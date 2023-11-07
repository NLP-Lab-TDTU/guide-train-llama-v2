from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from transformers import AutoTokenizer

# DEFAULT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}"""
DEFAULT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}{{ bos_token + '[INST] ' + '<<SYS>>\\n' + messages[0]['content'] + '\\n<</SYS>>\\n\\n' + messages[1]['content'] + ' [/INST]' }}{% for message in messages[:2] %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}{% else%}{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}{% endif %}"""

tokenizer = AutoTokenizer.from_pretrained("vietgpt/dama-2-7b-chat", verbose=False)
tokenizer.chat_template = DEFAULT_TEMPLATE
tokenizer.pad_token = '<pad>'

train_datasets = Dataset.from_dict({'text': []})
valid_datasets = Dataset.from_dict({'text': []})

list_datasets = [
    ("vietgpt/ultrachat", "train_2048"),
    ("vietgpt/orca", "train_2048"),
    ("vietgpt/WizardLM_evol_instruct_V2_196k", "train_2048"),
    
    ("vietgpt/databricks_dolly15k_en", "train"),
    ("vietgpt/databricks_dolly15k_vi", "train"),
    ("vietgpt/alpaca_en", "train"),
    ("vietgpt/alpaca_vi", "train"),

    ("vietgpt/grade_school_math", "train"),
    ("vietgpt/webglm-qa", "train"),
]

for dataset_name, dataset_split in list_datasets:
    train_dataset = load_dataset(dataset_name, split=f"{dataset_split}[1000:]")
    train_dataset = train_dataset.map(lambda x: {'text': tokenizer.apply_chat_template(x['messages'], tokenize=False)}, num_proc=64, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != 'text'])
    
    valid_dataset = load_dataset(dataset_name, split=f"{dataset_split}[:1000]")
    valid_dataset = valid_dataset.map(lambda x: {'text': tokenizer.apply_chat_template(x['messages'], tokenize=False)}, num_proc=64, remove_columns=valid_dataset.column_names)
    valid_dataset = valid_dataset.remove_columns([c for c in valid_dataset.column_names if c != 'text'])

    train_datasets = concatenate_datasets([train_datasets, train_dataset])
    valid_datasets = concatenate_datasets([valid_datasets, valid_dataset])
    
list_datasets = [
    ("vietgpt/basic", "train"),
]

for dataset_name, dataset_split in list_datasets:
    train_dataset = load_dataset(dataset_name, split=f"{dataset_split}")
    train_dataset = train_dataset.map(lambda x: {'text': tokenizer.apply_chat_template(x['messages'], tokenize=False)}, num_proc=64, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != 'text'])
    
    valid_dataset = load_dataset(dataset_name, split=f"{dataset_split}")
    valid_dataset = valid_dataset.map(lambda x: {'text': tokenizer.apply_chat_template(x['messages'], tokenize=False)}, num_proc=64, remove_columns=valid_dataset.column_names)
    valid_dataset = valid_dataset.remove_columns([c for c in valid_dataset.column_names if c != 'text'])

    train_datasets = concatenate_datasets([train_datasets, train_dataset])
    valid_datasets = concatenate_datasets([valid_datasets, valid_dataset])

datasets = DatasetDict({
    'train': train_datasets,
    'validation': valid_datasets,
})

datasets = datasets.shuffle(seed=42)

datasets.save_to_disk('my_sft_dataset', num_proc=128)
