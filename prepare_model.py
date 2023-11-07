import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", help="Name of the model", required=True)
parser.add_argument("--output-path", help="Path to save the model", required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("vietgpt/dama-2-7b-chat")
model = AutoModelForCausalLM.from_pretrained(args.model_name)

tokenizer.save_pretrained(args.output_path)
model.save_pretrained(args.output_path)
