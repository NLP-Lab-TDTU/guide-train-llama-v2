from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset

list_datasets = [
    "vietgpt/arxiv",
    "vietgpt/github",
    "vietgpt/open-web-math",
    "vietgpt/stackexchange",
    "vietgpt/CulturaX",
]

train_datasets = Dataset.from_dict({'text': []})

for dataset_name in list_datasets:
    train_dataset = load_dataset(dataset_name, split="train", num_proc=128)
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != 'text'])

    train_datasets = concatenate_datasets([train_datasets, train_dataset])

datasets = DatasetDict({
    'train': train_datasets,
})

datasets = datasets.shuffle(seed=42)

datasets.save_to_disk('my_dataset2', num_proc=128)
