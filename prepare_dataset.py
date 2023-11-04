from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset

list_datasets = [
    "vietgpt/openwebtext_en",
    "nlplabtdtu/webtext_cc_en",
    "nlplabtdtu/cc-2014-52",
    "nlplabtdtu/cc-2015-48",
    "nlplabtdtu/cc-2016-50",
    "nlplabtdtu/cc-2017-51",
    "nlplabtdtu/cc-2018-51",
    "nlplabtdtu/cc-2019-51",
    "nlplabtdtu/cc-2020-51",
    "nlplabtdtu/cc-2021-49",
    "nlplabtdtu/cc-2022-40",
    "vietgpt/the_pile_openwebtext2",
    "vietgpt/c4_vi",
    "vietgpt/OSCAR-2109",
    "vietgpt/OSCAR-2201",
    "vietgpt/OSCAR-2301"
]

train_datasets = Dataset.from_dict({'text': []})
valid_datasets = Dataset.from_dict({'text': []})

for dataset_name in list_datasets:
    train_dataset = load_dataset(dataset_name, split="train[1000:]", num_proc=128)
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != 'text'])
    
    valid_dataset = load_dataset(dataset_name, split="train[:1000]", num_proc=128)
    valid_dataset = valid_dataset.remove_columns([c for c in valid_dataset.column_names if c != 'text'])

    train_datasets = concatenate_datasets([train_datasets, train_dataset])
    valid_datasets = concatenate_datasets([valid_datasets, valid_dataset])

datasets = DatasetDict({
    'train': train_datasets,
    'validation': valid_datasets,
})

datasets = datasets.shuffle(seed=42)

datasets.save_to_disk('my_dataset1', num_proc=128)
