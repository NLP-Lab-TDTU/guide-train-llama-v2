### Requirements
CUDA version: 11.8

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets metrics tokenizers evaluate deepspeed
```

### Access Model Llama2
Due to the copyright of Llama 2, we have to set the model to private, in order to access it, please follow the steps below.
```bash
huggingface-cli login
# then paste token
```

### Prepare dataset

```bash
python prepare_dataset.py
```

### Prepare model

```bash
python prepare_model.py --model-name vietgpt/dama-2-7b-200000 --output-path /path/to/model
```

### Training

```
$ accelerate config
-----------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
-----------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?                                                                                                                                                                                                                   
multi-GPU
-----------------------------------------------------------------------------------------------------------------------
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
-----------------------------------------------------------------------------------------------------------------------                                                                                                                                
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
-----------------------------------------------------------------------------------------------------------------------                                                                                                                                
Do you want to use DeepSpeed? [yes/NO]: yes
-----------------------------------------------------------------------------------------------------------------------                                                                                                                                
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
-----------------------------------------------------------------------------------------------------------------------                                                                                                                                
What should be your DeepSpeed's ZeRO optimization stage?
2
-----------------------------------------------------------------------------------------------------------------------                                                                                                                                
Where to offload optimizer states?                                                                                                                                                                                                                     
none
-----------------------------------------------------------------------------------------------------------------------                                                                                                                                
Where to offload parameters?                                                                                                                                                                                                                           
none
-----------------------------------------------------------------------------------------------------------------------                                                                                                                                
How many gradient accumulation steps you're passing in your script? [1]: 16
-----------------------------------------------------------------------------------------------------------------------                                                                                                                                
Do you want to use gradient clipping? [yes/NO]: NO
-----------------------------------------------------------------------------------------------------------------------                                                                                                                                
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: NO
-----------------------------------------------------------------------------------------------------------------------                                                                                                                                
How many GPU(s) should be used for distributed training? [1]:8
-----------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
bf16
-----------------------------------------------------------------------------------------------------------------------                                                                                                                                
accelerate configuration saved at /home/cuong/.cache/huggingface/accelerate/default_config.yaml
```

Note: total batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus

You need to change some of the following values accordingly.
- num_processes: num_gpus. Notes: Set 8 to train on 8 GPUs. This command only works on 80GB GPUs
- model_name_or_path: Name of model. Default: [vietgpt/dama-2-7b-200000](https://huggingface.co/vietgpt/dama-2-7b-200000) (This is the original model of Llama 2 that has been converted to Huggingface format and replaced tokenizer part).
- tokenizer_name: Tokenizer of the model. Default: [vietgpt/dama-2-7b-200000](https://huggingface.co/vietgpt/dama-2-7b-200000) (This is the original model of Llama 2 that has been converted to Huggingface format and replaced tokenizer part).
- dataset_path: Path of the preprocessed dataset.

- per_device_train_batch_size. Default: 2
- per_device_eval_batch_size. Default: 2
- output_dir: Output directory.
- preprocessing_num_workers: Num processes to preprocess data. Default: 128 (Based on num_cpu)
- dataloader_num_workers: Num processes to load data for training. Default: 128 (Based on total batch_size or num_cpu)
- gradient_accumulation_steps
- logging_steps: Show log every n steps.
- save_steps: Save checkpoint every n steps.
- save_total_limit: Save only n latest checkpoints.

```bash
accelerate launch --multi_gpu --num_processes 8 run_clm.py \
--model_name_or_path /path/to/model \
--dataset_path ./my_dataset1 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--do_train --do_eval \
--output_dir checkpoints \
--preprocessing_num_workers 128 \
--torch_dtype bfloat16 \
--optim adafactor \
--dataloader_num_workers 64 \
--gradient_accumulation_steps 16 \
--logging_steps 100 \
--save_steps 500 \
--save_total_limit 10 \
--learning_rate 3e-5 \
--tokenizer_name /path/to/model |& tee -a train.log
```
