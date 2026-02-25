# Generate the predictions for direct evaluation. Assumes already marker-formatted datasets, and
# requires you to run generate_metrics.py separately to get the final evaluation scores.
 
import sys
import torch
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, HfArgumentParser, AutoModelForCausalLM
from pathlib import Path
from transformers import set_seed
from datasets import load_dataset
from utils.utils import NLLB_CODE, LANG_TABLE
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class EvalArguments:
    model_names_or_paths: str = field(
        metadata={"help": "Paths to the pre-trained model or model identifier from huggingface.co/models."}
    )
    output_dirs: str = field(
        metadata={"help": "Directory to save evaluation outputs."}
    )
    model_type: str = field(
        default="seq2seq",
        metadata={"help": "Type of model to use. Options are 'seq2seq', 'causal'."}
    )
    eval_data_path: str = field(
        default="evaluation_data/",
        metadata={"help": "Path to the parent evaluation data directory."}
    )
    eval_datasets: str = field(
        default="XQUAD,XQUAD_squarebracket,XQUAD_xml",
        metadata={"help": "Comma-separated list of evaluation datasets to use. This should match the names of the directories in eval_data_path."}
    )
    reverse: bool = field(
        default=False,
        metadata={"help": "If true, also do the reverse of source and target languages in the evaluation."}
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate during evaluation."}
    )
    max_source_length: int = field(
        default=512,
        metadata={"help": "Maximum length of the source sequences."}
    )
    num_beams: int = field(
        default=5,
        metadata={"help": "Number of beams for beam search during evaluation."}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Data type to use for PyTorch tensors. Options are 'float32', 'bfloat16', etc."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for evaluation."}
    )
    prompt_template: str = field(
        default="",
        metadata={"help": "Prompt template to use for causal models."}
    )

parser = HfArgumentParser(EvalArguments)
if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
    # If we pass only one argument to the script and it's the path to a yaml file,
    # let's parse it to get our arguments.
    args = parser.parse_yaml_file(Path(sys.argv[1]))[0]
else:
    args = parser.parse_args_into_dataclasses()[0]


# load tokenizer
set_seed(args.seed)

for model_name_or_path, output_dir in zip(args.model_names_or_paths.split(','), args.output_dirs.split(',')):
    print(f"Loading model from {model_name_or_path}")
    if args.model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=args.torch_dtype, device_map="auto")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, torch_dtype=args.torch_dtype, device_map="auto")
    model.eval()
    if "easyproject" in model_name_or_path:
        print("Using legacy behavior for tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", legacy_behaviour=True)
    elif args.model_type == "causal":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    for evaluation_dataset in args.eval_datasets.split(','):
        evaluation_dataset = evaluation_dataset.strip()
        eval_data_path = Path(args.eval_data_path) / evaluation_dataset
        for lang_dir in eval_data_path.iterdir():
            if lang_dir.is_dir():
                if args.reverse:
                    src, tgt = lang_dir.name.split('-')
                    lang_pairs = [f"{tgt}-{src}", lang_dir.name]
                else:
                    lang_pairs = [lang_dir.name]
                # reference path is independent of langpair order
                ref_path = lang_dir / f"test.{lang_dir.name}.json"
                for lang_pair in lang_pairs:
                    src_lang, tgt_lang = lang_pair.split('-')
                    
                    print(f"Evaluating {evaluation_dataset} for language pair {NLLB_CODE[src_lang]}, {NLLB_CODE[tgt_lang]}")
                    out_path = Path(output_dir) / "predictions" / f"{evaluation_dataset}.{lang_pair}.jsonl"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    dataset = load_dataset('json', data_files=str(ref_path))
                    tokenizer.src_lang, tokenizer.tgt_lang = NLLB_CODE[src_lang], NLLB_CODE[tgt_lang]
                    
                    def tokenize_function(examples):
                        inputs = [item[src_lang] for item in examples['translation']]
                        if args.model_type == "causal" and args.prompt_template:
                            inputs = [
                                tokenizer.apply_chat_template(
                                    [{
                                        "role": "user", 
                                        "content": args.prompt_template.format(src_lang=LANG_TABLE[src_lang], tgt_lang=LANG_TABLE[tgt_lang], text=inp)
                                        }],
                                    tokenize=False,
                                    add_generation_prompt=True
                                ) for inp in inputs]
                            return tokenizer(inputs, max_length=args.max_source_length, truncation=True, padding="max_length")
                        else:
                            return tokenizer(inputs, max_length=args.max_source_length, truncation=True, padding="max_length")
                    
                    dataset = dataset["train"].map(tokenize_function, batched=True, batch_size=args.batch_size).with_format("torch")
                    
                    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
                    predictions = []
                    for batch in tqdm(dataloader, desc=f"Evaluating {lang_pair}"):
                        with torch.no_grad():
                            outputs = model.generate(
                                input_ids=batch['input_ids'].cuda(),
                                attention_mask=batch['attention_mask'].cuda(),
                                max_new_tokens=args.max_new_tokens,
                                num_beams=args.num_beams,
                                forced_bos_token_id=tokenizer.convert_tokens_to_ids(NLLB_CODE[tgt_lang]) if args.model_type == "seq2seq" else None
                            )
                            if args.model_type == "causal":
                                outputs = outputs[:, batch['input_ids'].shape[1]:]  # remove input prompt
                        predictions.extend(outputs.cpu().numpy())
                    
                    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                    
                    # Save predictions to output file
                    with open(out_path, 'w', encoding='utf-8') as f:
                        for pred in predictions:
                            # remove any newlines or extra spaces
                            pred = pred.replace('\n', ' ').strip()
                            f.write(pred + '\n')
                    print(f"Predictions saved to {out_path}.")
                


