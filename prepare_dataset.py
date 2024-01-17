from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vietgpt/dama-2-7b", verbose=False)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
tokenizer.pad_token = '<pad>'

train_datasets = Dataset.from_dict({'text': []})

list_datasets = [
    # Code
    ("vietgpt/code_instructions_120k", "train"),
    ("vietgpt/WRN-Chapter-1", "train"),
    ("vietgpt/sql-create-context", "train"),
    ("vietgpt/glaive-code-assistant", "train"),
    ("vietgpt/Magicoder-OSS-Instruct-75K", "train"),
    ("vietgpt/Magicoder-Evol-Instruct-110K", "train"),
    ("vietgpt/Open-Platypus", "train"),

    # LM
    ("vietgpt/basic", "train"),
    ("vietgpt/lima", "train"),
    ("vietgpt/PIPPA", "train"),
    ("vietgpt/orca", "train_2048"),
    ("vietgpt/WizardLM_evol_instruct_V2_196k", "train_2048"),
    ("vietgpt/ultrachat", "train_2048"),
    ("vietgpt/argilla-ultrafeedback", "train"),
    ("vietgpt/webglm-qa", "train"),
    ("vietgpt/im-feeling-curious", "train"),
    ("vietgpt/arxiv-math-instruct-50k", "train"),
    ("vietgpt/arxiv-physics-instruct-tune-30k", "train"),
    ("vietgpt/arxiv_nlp_intstruct", "train"),
    ("vietgpt/arxiv-cs-ml-instruct-tune-50k", "train"),
    ("vietgpt/HC3-sft", "train"),
    ("vietgpt/no_robots", "train"),
    ("vietgpt/WildChat", "train"),
    ("vietgpt/EverythingLM-data-V3", "train"),
    ("vietgpt/databricks_dolly15k_en", "train"),
    ("vietgpt/databricks_dolly15k_vi", "train"),
    ("vietgpt/alpaca_en", "train"),
    ("vietgpt/alpaca_vi", "train"),

    # Math
    ("vietgpt/grade_school_math", "train"),
    ("vietgpt/MetaMathQA", "train"),
    ("vietgpt/goat", "train"),
    ("vietgpt/evol-codealpaca-v1", "train"),
    ("vietgpt/MathInstruct", "train"),

    # Vietnamese
    ("vietgpt/exam-vn-rationale", "train"),
    ("vietgpt/error-correction-vi", "train"),
    ("vietgpt/hdpl_sft", "train"),
    ("vietgpt/thuvienphapluat_sft", "train"),
    ("vietgpt/legal_citation_sft", "train"),
]

for dataset_name, dataset_split in list_datasets:
    train_dataset = load_dataset(dataset_name, split=dataset_split)
    train_dataset = train_dataset.map(lambda x: {'text': tokenizer.apply_chat_template(x['messages'], tokenize=False)}, num_proc=64, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != 'text'])
    train_datasets = concatenate_datasets([train_datasets, train_dataset])

datasets = DatasetDict({
    'train': train_datasets,
})

datasets = datasets.shuffle(seed=42)

datasets.save_to_disk('my_sft_dataset', num_proc=128)
