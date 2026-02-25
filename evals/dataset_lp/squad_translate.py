import json
import torch
import regex
from tqdm import tqdm
from pathlib import Path
from evals.dataset_lp.dataset_utils import insert_tags, has_token_stutter
from utils.utils import NLLB_CODE
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
from copy import deepcopy

INPUT_FILE = "./evaluation_data/squad/train-v1.1.json" 
MODEL_PATH = "thennal/nllb-200-3.3B-labelpigeon"
OUTPUT_PATH = "./outputs/labelpigeon"
TAG_TYPE = "xml" # "xml" or "squarebracket"
BATCH_SIZE = 64 # 64 for 25GB VRAM, 
MAX_LENGTH = 512

MAX_MISSING_ANSWERS = 1 # how many answers can be missing before we skip the paragraph, set to None to disable
NUM_SAMPLES = None  # corresponds to articles, not paragraphs (very roughly 40 paragraphs per article)
TGT_LANGS = ["de", "ar", "es", "hi", "vi", "zh"]

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)
if "easyproject" in MODEL_PATH:
    print("Using legacy behavior for tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", legacy_behaviour=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.src_lang = "eng_Latn"

def batch_translate(texts, target_lang, batch_size=BATCH_SIZE):
    out = []
    bos_id = tokenizer.convert_tokens_to_ids(target_lang)
    for i in tqdm(range(0, len(texts), batch_size)):
        chunk = texts[i:i+batch_size]
        inputs = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                forced_bos_token_id=bos_id,
                max_length=MAX_LENGTH,
                #num_beams=5,
                #repetition_penalty=1.1,  # repetition penalty to avoid repeating the same tokens
                #no_repeat_ngram_size=4,  # repetition blocking works better if this number is below num_beams  
                #renormalize_logits=True,  # recompute token probabilities after banning the repetitions
                #early_stopping=True,
            )
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out

def extract_tags(text, tag_type="xml"):
    if tag_type == "xml":
        soup = BeautifulSoup(text, "html.parser")
        tags = []
        for tag in soup.find_all():
            tags.append((tag.name, ' '.join(tag.text.split()).strip()))
        untagged_text = soup.get_text()
        # remove double spaces
        untagged_text = ' '.join(untagged_text.split())
        return tags, untagged_text
    elif tag_type == "squarebracket":
        pattern = r"\[(?:[^\[\]]|(?R))*\]"
        matches = regex.findall(pattern, text, overlapped=True)
        remove_brackets = lambda x: x.replace('[', '').replace(']', '').strip()
        return [('[]', remove_brackets(m[1:-1])) for m in matches], remove_brackets(text)

def fuzzy_match(a, b, threshold=0.5):
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() > threshold

# get the json file
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

# limit the data for testing
if NUM_SAMPLES:
    data["data"] = data["data"][:NUM_SAMPLES]

texts = []
questions = []
answers = []
for item in data["data"]:
    for paragraph in item["paragraphs"]:
        context = paragraph["context"]
        paragraph["marked_context"], paragraph["tag_map"] = insert_tags(context, paragraph["qas"], TAG_TYPE)
        texts.append(paragraph["marked_context"])
        for qas in paragraph["qas"]:
            questions.append((qas["question"], qas["id"]))
            answers.append((qas["answers"][0]["text"], qas["id"]))
    
for tgt_lang in TGT_LANGS:
    print(f"Translating to {tgt_lang}...")

    translated_texts = batch_translate(texts, NLLB_CODE[tgt_lang], BATCH_SIZE)
    translated_questions = batch_translate(
        [q[0] for q in questions], NLLB_CODE[tgt_lang], BATCH_SIZE
    )
    translated_question_map = {q[1]: tq for q, tq in zip(questions, translated_questions)}
    if TAG_TYPE == "squarebracket": # translate answers as well
        translated_answers = batch_translate(
            [a[0] for a in answers], NLLB_CODE[tgt_lang], BATCH_SIZE
        )
        translated_answer_map = {a[1]: ta for a, ta in zip(answers, translated_answers)}
    
    n_data = deepcopy(data)
    par_num = 0
    for item in n_data["data"]:
        for paragraph in item["paragraphs"]:
            tags, context = extract_tags(translated_texts[par_num], TAG_TYPE)
            paragraph["context"] = context
            # use the tags and the tag_map to assign new answers to the questions
            new_qas = []
            for tag_name, tag_text in tags:
                if TAG_TYPE == "xml": # for xml tags, search for the corresponding qas in the tag_map
                    if f"<{tag_name}>" in paragraph["tag_map"]:
                        qas_id, ans_id = paragraph["tag_map"][f"<{tag_name}>"]
                        for qas in paragraph["qas"]:
                            if qas["id"] == qas_id:
                                # only one answer per question anyway
                                if context.find(tag_text) == -1:
                                    print(f"Warning: Answer text '{tag_text}' not found in translated context for id {qas_id}. Skipping.")
                                    continue
                                new_qas.append({
                                    "id": qas_id,
                                    "question": translated_question_map[qas_id],
                                    "answers": [{
                                        "answer_start": context.find(tag_text),
                                        "text": tag_text
                                    }]
                                })
                elif TAG_TYPE == "squarebracket": # for square brackets, we find the closest answer matching the tag text
                    for qas in paragraph["qas"]:
                        if fuzzy_match(tag_text, translated_answer_map.get(qas["id"], "")):
                            new_qas.append({
                                "id": qas["id"],
                                "question": translated_question_map[qas["id"]],
                                "answers": [{
                                    "answer_start": context.find(tag_text),
                                    "text": tag_text
                                }]
                            })
                            # remove the question from the list to avoid duplicates
                            translated_answer_map.pop(qas["id"])
                            break
            # remove questions that have repetitions
            new_qas = [qas for qas in new_qas if not has_token_stutter(qas["question"])]
            # if there are too many missing answers, skip the paragraph (by setting qas to empty and filtering later)
            if (MAX_MISSING_ANSWERS is not None) and (len(new_qas) < len(paragraph["qas"]) - MAX_MISSING_ANSWERS):
                paragraph["qas"] = []
            else:
                paragraph["qas"] = new_qas
            par_num += 1
            # remove the tag_map and marked_context as it's no longer needed
            del paragraph["tag_map"]
            del paragraph["marked_context"]
    
    # remove all paragraphs that have no questions, or have repititions
    n_data["data"] = [
        {
            "paragraphs": [
                paragraph for paragraph in item["paragraphs"] if (paragraph["qas"] and not has_token_stutter(paragraph["context"]))
            ]
        }
        for item in n_data["data"]
    ]
    # save the translated data
    output_file = Path(OUTPUT_PATH) / "translated_squad" / f"train_{tgt_lang}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(n_data, f, indent=4, ensure_ascii=False)
    # output some info
    num_paragraphs = sum(len(item["paragraphs"]) for item in n_data["data"])
    num_questions = sum(len(paragraph["qas"]) for item in n_data["data"] for paragraph in item["paragraphs"])
    print(f"Translated {num_paragraphs} paragraphs and {num_questions} questions, saved to {output_file}.")