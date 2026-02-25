import torch
import json
import random
from tqdm import tqdm
from pathlib import Path
from itertools import product
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu.metrics import BLEU, CHRF, TER
from icu import BreakIterator, Locale, UnicodeString
from evals.dataset_lp.dataset_utils import insert_tags_from_spans, untag_text

MODEL_PATH = "thennal/nllb-200-3.3B-labelpigeon"
OUTPUT_PATH = "./outputs/labelpigeon"
TAG_TYPE = "xml" # xml, squarebracket, None
NUM_TAGS = "single" # "single", "multiple", "complex"
HYPERPARAMS_SET = {"p_close": [0.5], "p_open": [0.2]} # goes through every value in the lists
BATCH_SIZE = 256 
MAX_LENGTH = 256
BEAMS = 1 # generally not worth doing beam search
SEED = 0

if TAG_TYPE:
    MODE = f"{TAG_TYPE}-{NUM_TAGS}"
else:
    MODE = "notag"

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
if "easyproject" in MODEL_PATH:
    print("Using legacy behavior for tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", legacy_behaviour=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.src_lang = "eng_Latn"


def batch_translate(texts, target_lang, batch_size=16):
    if not texts:
        return []
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
                num_beams=BEAMS,
                #early_stopping=True,
            )
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out


def _geom_1_based(rng, p):
    k = 1
    while rng.random() > p:
        k += 1
    return k

import random
from icu import BreakIterator, Locale, UnicodeString

def generate_random_spans(text, flores_code, mode="single", rng=None, p_open=0.2, p_close=0.5):
    """
    Generates random spans to be inserted for synthetic labels, with the following modes:
      - "single": sample a single random contiguous span uniformly from all word-boundary spans
      - "multiple": at each word boundary, start a span w.p. p_open if none is open;
                    if open, end it w.p. p_close. Close any open span at sentence end.
                    No nesting, no overlaps.
      - "complex": at each word boundary, start a new span w.p. p_open (even if others are open);
                   for each open span, end it w.p. p_close. Close any remaining spans at sentence end.
                   Nested and overlapping spans are possible.
    """
    if rng is None:
        rng = random.Random()

    # tokenize into word spans (start,end char offsets)
    lang = flores_code.split("_")[0]
    bi = BreakIterator.createWordInstance(Locale(lang))
    u = UnicodeString(text)
    bi.setText(u)
    word_spans = []
    s = bi.first()
    for e in bi:
        tok = str(u[s:e])
        if tok.strip() and any(ch.isalnum() for ch in tok):
            word_spans.append((s, e))
        s = e

    n = len(word_spans)
    if n == 0:
        return []

    mode = mode.lower()

    # === SINGLE ===
    if mode == "single":
        length = _geom_1_based(rng, (1 - p_close))
        if length > n:
            length = n
        start_idx = rng.randint(0, n - length)
        return [(word_spans[start_idx][0], word_spans[start_idx + length - 1][1])]
    # === MULTIPLE ===
    elif mode == "multiple":
        spans = []
        open_span = None
        for t in range(n):
            if open_span is None:
                # no span active
                if rng.random() < p_open:
                    open_span = word_spans[t][0]  # char index start
            else:
                # span already open
                if rng.random() < p_close:
                    spans.append((open_span, word_spans[t][1]))
                    open_span = None
        # close if still open
        if open_span is not None:
            spans.append((open_span, word_spans[-1][1]))
        return spans

    # === COMPLEX ===
    elif mode == "complex":
        spans = []
        active = []  # list of (start_char)
        for t in range(n):
            # possibly open new span
            if rng.random() < p_open:
                active.append(word_spans[t][0])

            # update existing spans
            new_active = []
            for start_char in active:
                if rng.random() < p_close:
                    spans.append((start_char, word_spans[t][1]))
                else:
                    new_active.append(start_char)
            active = new_active
        # close remaining
        for start_char in active:
            spans.append((start_char, word_spans[-1][1]))
        return spans

    else:
        raise ValueError(f"Unknown mode: {mode}")


flores = load_dataset("Muennighoff/flores200", "all", trust_remote_code=True)['devtest']
langs = flores.column_names
langs = [lang.split('_', 1)[1] for lang in langs if lang.startswith('sentence_')]

hyperparam_names = sorted(HYPERPARAMS_SET.keys())
for hp_values in tqdm(
    [dict(zip(hyperparam_names, v)) for v in 
     product(*[HYPERPARAMS_SET[n] for n in hyperparam_names])],
    desc="Hyperparameter combinations"
):
    hp_str = "-".join(f"{k}{v}".replace('.', '') for k,v in hp_values.items())
    output_path = Path(OUTPUT_PATH) / "translated_flores200" / f"{MODE}-{hp_str}"
    #output_path = Path(OUTPUT_PATH) / "translated_flores200" / MODE
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Translating with hyperparameters: {hp_values}, output path: {output_path}")
    all_metrics = {}
    for lang in tqdm(langs):
        src_sents = flores['sentence_eng_Latn']
        if TAG_TYPE:
            # insert tags
            tagged_sents = []
            src_tags = []
            rng = random.Random(SEED)
            for sent in src_sents:
                spans = generate_random_spans(sent, "eng_Latn", mode=NUM_TAGS, rng=rng, **hp_values)
                tagged_sent, _ = insert_tags_from_spans(sent, spans, TAG_TYPE)
                tagged_sents.append(tagged_sent)
                src_tags.append(untag_text(tagged_sent, TAG_TYPE)[1])
            src_sents = tagged_sents

        translated_tagged = batch_translate(src_sents, lang, batch_size=BATCH_SIZE)
        if TAG_TYPE:
            # remove tags
            translated = []
            tl_tags = []
            for sent in translated_tagged:
                untagged_sent, tags = untag_text(sent, TAG_TYPE)
                translated.append(untagged_sent)
                tl_tags.append(tags)
        else:
            translated = translated_tagged
            
        metrics = {}
        metrics['bleu'] = BLEU().corpus_score(translated, [flores[f'sentence_{lang}']]).score
        metrics['chrf++'] = CHRF(word_order=2).corpus_score(translated, [flores[f'sentence_{lang}']]).score
        metrics['ter'] = TER().corpus_score(translated, [flores[f'sentence_{lang}']]).score
        
        # projection rate is the number of tags kept in the translation divided by the number of tags in the source
        # this is only relevant if we have tags
        if TAG_TYPE:
            # projection rate is the number of samples where the number and type of tags is preserved
            _sortt = lambda x: sorted([t[0] for t in x])
            metrics['projection_rate'] = sum([1 if _sortt(s) == _sortt(t) else 0 for s,t in zip(src_tags, tl_tags)]) / len(src_tags)
        all_metrics[lang] = metrics
        
        if TAG_TYPE:
            output = {
                "translations": [
                    {"source": src, "reference": ref, "translation": pred, "source_tagged": src_tagged, "translation_tagged": pred_tagged}
                        for src, ref, pred, src_tagged, pred_tagged in 
                        zip(src_sents, flores[f'sentence_{lang}'], translated, src_sents, translated_tagged)
                    ],
                    "metrics": metrics
                }
        else:
            output = {
                "translations": [
                    {"source": src, "reference": ref, "translation": pred} for src, ref, pred in 
                        zip(flores['sentence_eng_Latn'], flores[f'sentence_{lang}'], translated)
                    ],
                    "metrics": metrics
                }
        output_file = output_path / f"{lang}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    # write overall metrics
    all_metrics["average"] = {
        "bleu": sum(m["bleu"] for m in all_metrics.values()) / len(all_metrics),
        "chrf++": sum(m["chrf++"] for m in all_metrics.values()) / len(all_metrics),
        "ter": sum(m["ter"] for m in all_metrics.values()) / len(all_metrics),
        "projection_rate": sum(m.get("projection_rate", 0) for m in all_metrics.values()) / len(all_metrics) if TAG_TYPE else None
    }
    with open(output_path / "all_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)
        
            