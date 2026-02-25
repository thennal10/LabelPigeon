import re
import torch
import jieba
from pathlib import Path
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from evals.dataset_lp.dataset_utils import has_token_stutter
from utils.utils import NLLB_CODE, LANG_TABLE
from datasets import load_dataset


# === Paths & constants ===
MODEL_PATH = "thennal/nllb-200-3.3B-labelpigeon"
OUTPUT_PATH = "./outputs/labelpigeon"
MAX_LENGTH = 256
DOWNSTREAM_MAX_LENGTH = 256  # downstream length filter
BATCH_SIZE = 32  # translation batch size
TAG_TYPE = "xml"  # "squarebracket" or "xml"
MODEL_TYPE = "seq2seq"  # "seq2seq" or "causal"
PROMPT_TEMPLATE = "Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {text}\n{tgt_lang}: " # prompt template for causal models


# UNER target languages (translate from English -> XX)
TARGET_LANGS = ['ceb', 'da', 'de', 'hr', 'pt', 'ru', 'sk', 'sr', 'sv', 'tl', 'zh']

NER_TAGS = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}  # we will ignore MISC tags

idx_to_tag = {v: k for k, v in NER_TAGS.items()}
start_tags = ['B-PER', 'B-ORG', 'B-LOC']
continue_tags = ['I-PER', 'I-ORG', 'I-LOC']

# Length filter tokenizer (just for a rough cap, like in your CoNLL script)
downstream_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")

# NLTK tokenization / detokenization
nltk_tokenizer = TreebankWordTokenizer()
nltk_detokenizer = TreebankWordDetokenizer()

# Load model + tokenizer
if MODEL_TYPE == "causal":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
if "easyproject" in MODEL_PATH:
    print("Using legacy behavior for tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", legacy_behaviour=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.src_lang = "eng_Latn"

# Output dir
output_root = Path(OUTPUT_PATH) if OUTPUT_PATH else Path(MODEL_PATH)
output_dir = output_root / "translated_uner"

# insert xml tags according to the NER tags
def tag_ner(example):
    sent_ls = []
    ner_map = []
    xml_count = 0
    for i, (token, ner_tag_i) in enumerate(zip(example['tokens'], example['ner_tags'])):
        ner_tag = idx_to_tag[ner_tag_i]
        # check if its the relevant tags
        if ner_tag in start_tags + continue_tags:
            word = ''
            # start a new tag if it is a start tag
            if ner_tag in start_tags:
                if TAG_TYPE == "squarebracket":
                    word += f"[{token}"
                    ner_map.append((token, ner_tag)) # if squarebracket, map token to tag
                else:  # xml
                    xml_tag = chr(ord('a') + xml_count)
                    xml_count +=  1
                    ner_map.append((xml_tag, ner_tag)) # if xml, map xml_tag to ner_tag
                    word += f"<{xml_tag}>{token}"
            else: # continuation tag
                word += token
            # if final eleement or the next tag is not a continuation tag, close the current tag
            if (i + 1 == len(example['tokens'])) or (idx_to_tag[example['ner_tags'][i+1]] not in continue_tags): 
                if TAG_TYPE == "squarebracket":
                    word += "]"
                else:  # xml
                    word += f"</{xml_tag}>"
            sent_ls.append(word)
        else:
            sent_ls.append(token)
    sent = nltk_detokenizer.detokenize(sent_ls)
    example['sentence'] = sent.strip()
    example['map'] = ner_map
    return example

def fuzzy_match(a, b, threshold=0.5):
    return SequenceMatcher(None, a, b).ratio() >= threshold

# convert translated XML-tagged sentences to CoNLL format
def xml_ner_to_conll(entry, lang):
    # extract tokens + NER spans
    
    chunks = []  # list of (text, tag)
    if TAG_TYPE == "squarebracket": # this assumes no nested tags, which is true for NER
        parts = re.split(r'\[(.*?)\]', entry['translated_sentence'])
        src_tags_and_contents = entry['map']  # list of (tag, content)
        # Assign each hypothesis tag to the corresponding source tag using SequenceMatcher
        for i, p in enumerate(parts):
            if not p.strip():
                continue # skip empty parts
            if i % 2 == 1: # odd elements are inside square brackets
                # just find the first match above 0.5
                for src_content, src_tag in src_tags_and_contents:
                    if fuzzy_match(p.strip(), src_content, 0.5):
                        chunks.append((p, src_tag))
                        src_tags_and_contents.remove([src_content, src_tag])  # remove to avoid duplicates (hf swaps tuples to lists)
                        break
                else: # if not found, return empty to skip this example
                    return []
            else:
                chunks.append((p, 'O'))
    else: # xml
        soup = BeautifulSoup(entry['translated_sentence'], 'html.parser')
        tag_to_label = dict(entry['map'])
        
        for element in soup.contents:
            if isinstance(element, str):
                chunks.append((element, 'O'))
            else:
                tag = element.name
                ner_label = tag_to_label.get(tag, 'O')
                chunks.append((element.get_text(), ner_label))

        # check if all tags in the mapping exist in the chunks
        for xml_tag, ner_tag in tag_to_label.items():
            if ner_tag not in [c[1] for c in chunks]:
                print(f"Warning: Tag '{xml_tag}' not found in translated sentence: {entry['translated_sentence']}")
                return []

        
    # tokenize and assign BIO tags
    conll_output = []
    for text_chunk, ner_tag in chunks:
        if lang == "zh":
            tokens = list(jieba.cut(text_chunk))
        else:
            tokens = nltk_tokenizer.tokenize(text_chunk)

        if ner_tag == 'O':
            conll_output.extend([(tok, 'O') for tok in tokens])
        else:
            try:
                conll_output.append((tokens[0], ner_tag))  # B-XXX
                conll_output.extend([(tok, 'I' + ner_tag[1:]) for tok in tokens[1:]])  # I-XXX
            except IndexError:
                print(f"Warning: Empty chunks found in the translated sentence: {entry['translated_sentence']}")
                return []
    return conll_output

# translate with model
def translate(examples, model_type="seq2seq", src_lang="en", tgt_lang=None):
    if model_type == "seq2seq":
        inputs = tokenizer(examples, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(model.device)
    elif model_type == "causal":
        inputs = [
            tokenizer.apply_chat_template(
                [{
                    "role": "user", 
                    "content": PROMPT_TEMPLATE.format(src_lang=LANG_TABLE[src_lang], tgt_lang=LANG_TABLE[tgt_lang], text=inp)
                    }],
                tokenize=False,
                add_generation_prompt=True
            ) for inp in examples]
        inputs = tokenizer(inputs, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding="max_length").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(NLLB_CODE[tgt_lang]) if model_type == "seq2seq" else None, 
            max_new_tokens=MAX_LENGTH,
        ) #, num_beams=5)
        if model_type == "causal":
            outputs = outputs[:, inputs['input_ids'].shape[1]:]  # remove input prompt

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    

# ----- Main dataset load & processing -----

# Load English UNER (config 'en'); typical splits: train/validation/test (names may vary, we iterate generically)
uner_en = load_dataset("universalner/universal_ner", "en_ewt", split="train", trust_remote_code=True)

# Insert XML tags (works per-row), and only take the first 1000 examples for testing
uner_en = uner_en.map(tag_ner, batched=False)

# For each target language, translate + reconstruct + write files
for i, lang in enumerate(TARGET_LANGS):
    print(f"Translating UNER EN -> {lang} ({i+1}/{len(TARGET_LANGS)})...")
    
    if TAG_TYPE == "squarebracket":
        # translate the mapping as well for squarebracket
        uner_en = uner_en.map(lambda x: {"map": [(translate(content, model_type=MODEL_TYPE, src_lang="en", tgt_lang=lang)[0], tag) for content, tag in x["map"]]}, batched=False)
    
    # Translate in batches
    translated = uner_en.map(lambda x: {"translated_sentence": translate(x["sentence"], model_type=MODEL_TYPE, src_lang="en", tgt_lang=lang)}, batched=True, batch_size=BATCH_SIZE)

    # Length filter using downstream tokenizer
    translated = translated.filter(
        lambda x: len(downstream_tokenizer(x["translated_sentence"])["input_ids"]) <= DOWNSTREAM_MAX_LENGTH
    )

    lang_dir = output_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    out_path = lang_dir / f"train.txt"
    total_examples = len(translated)
    kept = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in translated:
            # Optional stutter filter
            if has_token_stutter(ex["translated_sentence"]):
                continue

            conll_output = xml_ner_to_conll(ex, lang)
            if not conll_output:
                continue

            kept += 1
            for token, tag in conll_output:
                f.write(f"{token} {tag}\n")
            f.write("\n")

    print(f"Wrote {kept} / {total_examples} examples to {out_path}")
