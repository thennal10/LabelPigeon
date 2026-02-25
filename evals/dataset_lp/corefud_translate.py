import re, torch
import unicodedata
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString, Tag
from nltk.tokenize.treebank import TreebankWordTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from difflib import SequenceMatcher
from utils.utils import NLLB_CODE

# ====== config ======
ONTONOTES_EN = "./ontonotes-release-5.0/data/files/data/english/annotations"
MODEL_PATH = "thennal/nllb-200-3.3B-labelpigeon"
OUTPUT_ROOT = "./outputs/labelpigeon/translated_ontonotes"
TARGET_LANGS = ["de", "he", "ca", "cs", "fr", "gr", "hi", "hu",
                "ko", "lt", "no", "cu", "pl", "ru", "es", "tr"]
MAX_LEN = 256
DOWNSTREAM_MAX_LEN = 256
BATCH_SIZE = 64
MAX_SENTS = 6 # max sentences per document to process

TAG_TYPE = "xml" # "squarebracket" or "xml"
REMAP_COREF_IDS = True  # whether to remap coref ids to be unique per document (seems like it's needed for OntoNotes)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
if "easyproject" in MODEL_PATH:
    print("Using legacy behavior for tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", legacy_behaviour=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.src_lang = "eng_Latn"

downstream_tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
tb = TreebankWordTokenizer()


def sanitize(text):
    _ZW = dict.fromkeys([0x200B, 0x200C, 0x200D, 0xFEFF], None)    
    PLACEHOLDER_RE = re.compile(r"""
    (?:                           # one of...
        \*(?:PRO|T|EXP|ICH)\*(?:-\d+)?  # *PRO*  or *T* or *EXP* with optional -digits
      | \*-\d+                      # bare *-1, *-2, ...
      | \[UNK\]                     # [UNK]
      | %                           # stray percent
      | \bpw\b                      # the literal token 'pw'
    )
    """, re.X)

    if text is None:
        return ""

    # 0) remove common zero-widths
    _ZW = dict.fromkeys([0x200B, 0x200C, 0x200D, 0xFEFF], None)

    s = text.translate(_ZW)
    s = unicodedata.normalize("NFKC", s)

    # normalize quotes/backticks → straight ASCII quotes
    s = s.replace("“", '"').replace("”", '"').replace("„", '"')
    s = s.replace("‘", "'").replace("’", "'")
    s = re.sub(r"`\s*`", '"', s)   # ``  → "
    s = s.replace("`", "'")        # lone backticks → '

    # collapse silly runs of quotes
    s = re.sub(r"'{3,}", "''", s)
    s = re.sub(r'"{3,}', '""', s)

    # *** remove inline parser artifacts ***
    s = PLACEHOLDER_RE.sub("", s)

    # normalize whitespace, drop C1 controls
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r'[\u0080-\u009F]', '', s)
    return s

def batch_translate(texts, target_lang, batch_size=64):
    out = []
    bos_id = tokenizer.convert_tokens_to_ids(target_lang)
    for i in tqdm(range(0, len(texts), batch_size)):
        chunk = texts[i:i+batch_size]
        inputs = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                forced_bos_token_id=bos_id,
                max_length=MAX_LEN,
                #num_beams=5,
                #repetition_penalty=1.1,  # repetition penalty to avoid repeating the same tokens
                #no_repeat_ngram_size=4,  # repetition blocking works better if this number is below num_beams  
                #renormalize_logits=True,  # recompute token probabilities after banning the repetitions
                #early_stopping=True,
            )
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out

def renametags_and_map(text):
    tagmap = {}
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all():
        if TAG_TYPE == "xml":
            # swap the tag to <a>, <b>, etc.
            if len(tagmap) >= 26:
                print("Warning: more than 26 tags in a sentence.")
                return [], {}
            new_tag = chr(ord('a') + len(tagmap))
            tagmap[new_tag] = tag.attrs
            tag.replace_with(soup.new_tag(new_tag, string=tag.text))
        elif TAG_TYPE == "squarebracket":
            inner = tag.text
            tagmap[inner] = tag.attrs
            tag.replace_with(f"[{inner}]")
    
    return str(soup), tagmap    
        

def iter_tokens_with_brackets(sentence, tagmap, lang):

    if TAG_TYPE == "squarebracket":
        # tagmap: {inner_text: {"id": "...", "type": "..."}}
        stream = []
        remaining = list(tagmap.items())  # [(text, attrs), ...]
        stack = []                        # [{'start': int, 'text': str}, ...]
        spans = []                        # [{'start': int, 'end': int, 'text': str}, ...]
        buf = ""

        def emit_text(s):
            if not s:
                return
            # accumulate literal text into all open spans' text (for nesting)
            for fr in stack:
                fr['text'] += s
            toks = tb.tokenize(s)
            for tok in toks:
                stream.append((tok, [], []))  # (token, opens, closes)

        # single pass scanner with bracket stack
        for ch in sentence:
            if ch == '[':
                emit_text(buf); buf = ""
                stack.append({'start': len(stream), 'text': ""})
            elif ch == ']':
                emit_text(buf); buf = ""
                if not stack:
                    # unmatched ']' -> treat as literal
                    emit_text(']')
                    continue
                fr = stack.pop()
                spans.append({'start': fr['start'], 'end': len(stream), 'text': fr['text'].strip()})
            else:
                buf += ch
        emit_text(buf)

        if stack:
            # unclosed '['
            print(f"Warning: {len(stack)} unclosed '[' in sentence, ignoring them.")
            for fr in stack:
                emit_text('[' + fr['text'])
            stack = []
            

        # Assign each span to the first fuzzy-matching entry from tagmap (≥ 0.5), greedily
        for sp in spans:
            chosen = None
            for idx, (src_text, attrs) in enumerate(remaining):
                if SequenceMatcher(None, sp['text'], src_text.strip()).ratio() >= 0.5:
                    chosen = (idx, attrs)
                    break
            if chosen is None:
                continue
            idx, attrs = chosen
            del remaining[idx]

            cid = attrs.get("id")
            if cid is None:
                continue
            ctyp = attrs.get("type")
            e_id = f"e{cid}"
            etype = "1"
            open_marker = (f"({e_id}-{etype}-1-CorefType:{ctyp}" if ctyp else f"({e_id}-{etype}-1")
            is_single = (sp['end'] - sp['start'] == 1)
            if is_single:
                stream[sp['start']][1].append(open_marker + ")")
            else:
                stream[sp['start']][1].append(open_marker)
                stream[sp['end'] - 1][2].append(f"{e_id})")

        # sentence without brackets
        return stream, re.sub(r'[\[\]]', '', sentence)
                    
    elif TAG_TYPE == "xml":    
        stream = []
        soup = BeautifulSoup(sentence, "html.parser")
    
        def emit_text(s):
            toks = tb.tokenize(s)
            for tok in toks:
                stream.append((tok, [], []))  # (token, opens, closes)

        def walk(node):
            if isinstance(node, NavigableString):
                s = str(node)
                if s.strip():
                    emit_text(s)
                return
            if isinstance(node, Tag):
                name = node.name
                info = tagmap.get(name, {})
                cid, ctyp = info.get("id"), info.get("type")
                if cid is None:
                    print(f"Warning: tag has no id attribute, skipping.")
                    # just skip this one
                    for ch in node.children:
                        walk(ch)
                    return
                before_len = len(stream)
                for ch in node.children:
                    walk(ch)
                after_len = len(stream)
                if after_len == before_len:
                    return

                e_id = f"e{cid}"
                etype = "1"
                open_marker = (
                    f"({e_id}-{etype}-1-CorefType:{ctyp}"
                    if ctyp else
                    f"({e_id}-{etype}-1"
                )

                is_single_token = (after_len - before_len == 1)
                if is_single_token:
                    # put the closer directly on the opener: "(...)"  (no trailing eID) 
                    stream[before_len][1].append(open_marker + ")")
                else:
                    stream[before_len][1].append(open_marker)
                    # multi-token: close with id on the last token
                    stream[after_len - 1][2].append(f"{e_id})")
                return

        for child in soup.contents:
            walk(child)
        return stream, soup.get_text()


coref_files = sorted(Path(ONTONOTES_EN).glob("*/*/*/*.coref"))  # e.g. bn/cnn/01/*.coref
#coref_files = coref_files[:5]  # DEBUG: limit to first 50 files
print(f"Found {len(coref_files)} coref files to process")

sents, tagmaps, docsplits = [], [], [0]
global_tag_id = 0
for f in coref_files:
    with open(f, "r", encoding="utf-8") as f:
        raw = f.read()
    # extract DOCNO for sent ids
    m = re.search(r'DOCNO="([^"]+)"', raw)
    docno = (m.group(1) if m else Path(f).stem).replace("/", "-")
    # keep only TEXT blocks
    texts = re.findall(r"<TEXT[^>]*>(.*?)</TEXT>", raw, flags=re.S)
    if not texts: continue

    # prepare sentences with short tags + a global tag map per sentence
    num_sents = sum(1 for block in texts for sent in re.split(r"\n+", block) if sent.strip())
    if MAX_SENTS and num_sents > MAX_SENTS:
        continue
    
    for block in texts:
        for sent in re.split(r"\n+", block):
            if not sent.strip(): continue
            nsent, tagmap = renametags_and_map(sanitize(sent))
            sents.append(nsent)
            tagmaps.append(tagmap)
    
    if REMAP_COREF_IDS:
        unique_ids = set((tag_attrs['id'] for tagmap in tagmaps[-num_sents:] for tag_attrs in tagmap.values()))
        id_map = {old_id: str(i+global_tag_id) for i, old_id in enumerate(sorted(unique_ids))}
        for tagmap in tagmaps[-num_sents:]:
            for tag_attrs in tagmap.values():
                old_id = tag_attrs['id']
                tag_attrs['id'] = id_map[old_id]
        global_tag_id += len(unique_ids)
    docsplits.append(num_sents + docsplits[-1])

print(f"Prepared {len(sents)} sentences in {len(docsplits)-1} documents.")
for lang in tqdm(TARGET_LANGS):
    print(f"Processing language {lang}...")
    print(sents[:3])
    trans_sents = batch_translate(sents, NLLB_CODE[lang], batch_size=BATCH_SIZE)
    trans_sents = [sanitize(s) for s in trans_sents]
    print(trans_sents[:3])
    print(tagmaps[:3])
    if TAG_TYPE == "squarebracket":
        # translate tagmap keys too
        translated_tagmap_keys = batch_translate([k for tm in tagmaps for k in tm.keys()], NLLB_CODE[lang], batch_size=BATCH_SIZE)
        idx = 0
        for tm in tagmaps:
            for old_key in list(tm.keys()):
                new_key = translated_tagmap_keys[idx]
                idx += 1
                if new_key != old_key:
                    tm[new_key] = tm[old_key]
                    del tm[old_key]
    print(tagmaps[:3])
    # split translations by docsplits
    doc_sents = [trans_sents[docsplits[i]:docsplits[i+1]] for i in range(len(docsplits)-1)]
    doc_tagmaps = [tagmaps[docsplits[i]:docsplits[i+1]] for i in range(len(docsplits)-1)]
    
    out_lang_dir = Path(OUTPUT_ROOT) / lang / f"{lang}-train.conllu"
    out_lang_dir.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_lang_dir, "w", encoding="utf-8") as outf:
        for j, (tsents, tmaps) in enumerate(zip(doc_sents, doc_tagmaps)):
            outf.write(f"# newdoc id = doc-{j+1}\n")
            #outf.write(f"# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC\n")
            outf.write(f"# global.Entity = eid-etype-head-other\n")
            for i, (sent, tmap) in enumerate(zip(tsents, tmaps)):
                tokens_with_marks, untagged_sent = iter_tokens_with_brackets(sent, tmap, lang)
                outf.write(f"# sent_id = {i+1}\n")
                outf.write(f"# text = {untagged_sent}\n")
                for i, (tok, opens, closes) in enumerate(tokens_with_marks, start=1):
                    if not str(tok).strip():
                        print(f"Warning: empty token in doc {j+1} sent {i}, skipping.")
                        if (opens or closes):
                            print(f"  but has brackets: {' '.join(opens)} ... {' '.join(closes)}")
                        continue
                    misc = "_"
                    if opens or closes:
                        entity = "".join(opens) + "".join(closes)
                        misc = f"Entity={entity}"
                    outf.write(f"{i}\t{tok}\t_\t_\t_\t_\t0\t_\t_\t{misc}\n")
                outf.write("\n")
    
    print(f"Wrote out {len(docsplits)} documents and {len(trans_sents)} sentences to {out_lang_dir}.")