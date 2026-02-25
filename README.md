# LabelPigeon

**Just Use XML: Revisiting Joint Translation and Label Projection**

LabelPigeon is a simple yet effective approach for jointly performing machine translation and cross-lingual label projection using XML tags. By fine-tuning NLLB-200 3.3B on XML-tagged parallel data, LabelPigeon transfers span-level annotations (entities, arguments, mentions, etc.) across languages while maintaining or improving translation quality in a single forward pass.

## Key Features

- **Joint Translation & Label Projection**: Translate text and project labels simultaneously with no additional computational overhead
- **XML-Based Markers**: Use XML tags for direct span correspondence, gracefully handling nested and overlapping annotations
- **Improved Translation Quality**: Achieves better COMET scores compared to baseline models and marker-based methods

## Installation

Built with Python 3.11. We recommend [uv](https://docs.astral.sh/uv/) for installing dependencies.
```bash
git clone https://github.com/yourusername/LabelPigeon.git
cd LabelPigeon
uv sync # alternatively, use 'pip install -r requirements.txt'
```

## Quick Start

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("thennal/nllb-200-3.3B-labelpigeon")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")

# Prepare tagged input (source language: English, target: German)
source_text = "The <a>Eiffel Tower</a> is located in <b>Paris</b>, <c>France</c>."
inputs = tokenizer(source_text, return_tensors="pt", src_lang="eng_Latn")

# Translate with label projection
outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["deu_Latn"])
translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
```
## Evaluation

### Dataset Converters

Convert existing datasets to XML-tagged format for training or evaluation:

```bash
# Convert MLQA dataset
python -m dataset_converters.mlqa_convert \
  --en_source_path ./MLQA/MLQA_V1/test/test-context-en-question-en.json \
  --test_folder_path ./MLQA/MLQA_V1/test \
  --tag_type xml \
  --languages en,de,ar,es,vi,hi,zh \
  --out_dir ./evaluation_data

# Convert XQuAD dataset
python -m dataset_converters.xquad_convert \
  --en_source_path ./xquad/xquad.en.json \
  --test_folder_path ./xquad/ \
  --tag_type squarebracket \
  --languages ar,de,el,en,es,hi,ro,ru,th,tr,vi,zh \
  --out_dir ./evaluation_data

# Convert Salesforce XML dataset
python -m dataset_converters.salesforce_xml_convert \
  --input_dir ./salesforce-data \
  --languages ende,enfr,enja \
  --out_dir ./training_data \
  --tag_type alphabetic \
  --remap True
```

### Direct Label Projection Evaluation

Evaluate label projection accuracy on parallel annotated datasets, after converting:

```bash
# Run predictions
python -m evals.direct_eval configs/eval/lp_direct_eval.yaml

# Generate metrics
python -m evals.generate_metrics_direct configs/eval/gen_metrics.yaml
```

### Translation Quality Evaluation

```bash
python -m evals.translation_eval configs/eval/lp_translate_eval.yaml
```

### Downstream Evaluation

Downstream evaluation depends on external data and repositories. Sample scripts for converting datasets are provided in `evals/dataset_lp`.
## Training

Fine-tune your own LabelPigeon model on XML-tagged data, after converting:

```
python run_llmmt.py configs/training/nllb.yaml
```

## Project Structure

```
LabelPigeon/
├── dataset_converters/           # Scripts to convert datasets to XML format
│   ├── mlqa_convert.py          
│   ├── xquad_convert.py           
│   └── salesforce_xml_convert.py 
│
├── evals/                        # Evaluation scripts and metrics
│   ├── direct_eval.py           # Direct label projection evaluation
│   ├── generate_metrics_direct.py # Metrics generation for direct evaluation
│   ├── translation_eval.py      # Translation quality evaluation (BLEU, chrF++, COMET)
│   └── dataset_lp/              # Downstream dataset translation scripts
│
├── utils/                        # Core utilities and helpers
├── configs/                      # Configuration files (YAML)
│   ├── training/                # Training configurations
│   └── eval/                    # Evaluation configurations
│
├── evaluation_data/              # Processed evaluation datasets
├── training_data/                # Processed training datasets
├── outputs/                      # Model outputs and evaluation results
│
├── run_llmmt.py                 # Training script
│
├── requirements.txt             
├── pyproject.toml               
└── README.md                    
```

