### Access Model Llama2
Due to the copyright of Llama 2, we have to set the model to private, in order to access it, please follow the steps below.
```bash
huggingface-cli login
# then paste token
```

### Training
Note: total batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpu

You need to change some of the following values accordingly.
- model_name_or_path: Name of model. Default: [vietgpt/llama-2-7b-original](https://huggingface.co/vietgpt/llama-2-7b-original) (This is the original model of Llama 2 that has been converted to Huggingface format).
- tokenizer_name: Tokenizer of model. Default: [vietgpt/llama-2-7b-original](https://huggingface.co/vietgpt/llama-2-7b-original) (This is the original model of Llama 2 that has been converted to Huggingface format).

- dataset_name: Name of dataset. Get dataset [here](https://huggingface.co/datasets?task_categories=task_categories:text-generation&task_ids=task_ids:language-modeling)

- per_device_train_batch_size. Default: 1
- per_device_eval_batch_size. Default: 1
- output_dir: Ouput directory.
- preprocessing_num_workers: Num processes to preprocess data. Default: 128 (Base on num_cpu)
- dataloader_num_workers: Num processes to load data for training. Default: 64 (Base on total batch_size)
- gradient_accumulation_steps
- logging_steps: Show log every n steps.
- save_steps: Save checkpoint every n steps.
- save_total_limit: Save only n latest checkpoints.

```bash
python run_clm.py \
--model_name_or_path vietgpt/llama-2-7b-original \
--dataset_name EleutherAI/pile \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--do_train --do_eval \
--output_dir checkpoints \
--preprocessing_num_workers 128 \
--torch_dtype bfloat16 \
--optim adafactor \
--dataloader_num_workers 64 \
--gradient_accumulation_steps 32 \
--logging_steps 10 \
--save_steps 10 \
--save_total_limit 10 \
--tokenizer_name vietgpt/llama-2-7b-original
```
