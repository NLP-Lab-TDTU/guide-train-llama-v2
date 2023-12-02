import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--model-repo", required=True)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_path)

model.push_to_hub(args.model_repo, private=True)
