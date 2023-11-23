import os
from argparse import ArgumentParser
from itertools import chain

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset

def parse_args():
    args = ArgumentParser()
    args.add_argument('--datasets', type=str, default="vietgpt/arxiv,vietgpt/github,vietgpt/open-web-math,vietgpt/stackexchange")
    args.add_argument('--tokenizer_name', type=str, default='vietgpt/dama-2-7b')
    args.add_argument('--output_dir', type=str, default='processed_data')
    args.add_argument('--block_size', type=int)
    args.add_argument('--overwrite_cache', action='store_true')
    args.add_argument('--num_proc', type=int, default=os.cpu_count() // 2)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--test', action='store_true')
    return args.parse_args()

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # prepare datasets
    list_datasets = args.datasets.split(',')

    train_datasets = Dataset.from_dict({'text': []})

    stats = {
        'datasets': {},
        'num_tokens': {},
        'selected_idx': {},
    }

    for dataset_name in list_datasets:
        if args.test:
            temp_dataset = load_dataset(dataset_name, split="train[:10000]", num_proc=args.num_proc)
        else:
            temp_dataset = load_dataset(dataset_name, split="train", num_proc=args.num_proc)

        temp_dataset = temp_dataset.remove_columns([c for c in temp_dataset.column_names if c != "text"])
        temp_dataset = temp_dataset.map(lambda x: {'num_tokens': len(tokenizer.tokenize(x['text']))}, num_proc=args.num_proc, remove_columns=['text'])

        stats['datasets'][dataset_name] = temp_dataset
        stats['num_tokens'][dataset_name] = sum(temp_dataset['num_tokens'])
        stats['selected_idx'][dataset_name] = 0

    total_num_tokens = sum(stats['num_tokens'].values())
    mean_num_tokens = total_num_tokens // len(list_datasets)
    print('total_num_tokens', total_num_tokens)
    print('mean_num_tokens', mean_num_tokens)

    for dataset_name in list_datasets:
        idx = 0
        total_tokens = 0
        while total_tokens < mean_num_tokens and idx < len(stats['datasets'][dataset_name]):
            total_tokens += stats['datasets'][dataset_name][idx]['num_tokens']
            idx += 1
        stats['selected_idx'][dataset_name] = idx

    for dataset_name in list_datasets:
        if args.test:
            train_dataset = load_dataset(dataset_name, split="train[:10000]", num_proc=args.num_proc)
        else:
            idx = stats['selected_idx'][dataset_name]
            train_dataset = load_dataset(dataset_name, split="train[:%d]" % idx, num_proc=args.num_proc)
        train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != "text"])
        train_datasets = concatenate_datasets([train_datasets, train_dataset])

    raw_datasets = DatasetDict({
        'train': train_datasets,
    })

    raw_datasets = raw_datasets.shuffle(seed=args.seed)

    # tokenize datasets
    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        # clm input could be much much longer than block_size
        return output

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=["text"],
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    block_size = 2048

    # group texts in blocks of block_size
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_proc,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    if os.path.exists(args.output_dir) and args.overwrite_cache:
        os.remove(args.output_dir)

    lm_datasets.save_to_disk(args.output_dir)

if __name__ == '__main__':
    main()
