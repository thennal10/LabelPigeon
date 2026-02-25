import json
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer
from argparse import ArgumentParser

parser = ArgumentParser(description="Convert XQuAD dataset to JSON format with optional tagging.")
parser.add_argument("--en_source_path", type=str, default="./xquad/xquad.en.json", help="Path to English source file")
parser.add_argument("--test_folder_path", type=str, default="./xquad/", help="Path to test folder containing language files")
parser.add_argument("--max_length", type=int, default=452, help="Maximum length for untagged context (512 - 60)")
parser.add_argument("--max_length_tokenized", type=int, default=512, help="Maximum tokenized length,  used for checking if the tagged context exceeds this length")
parser.add_argument("--tag_type", type=str, default="squarebracket", choices=["squarebracket", "xml", None], help="Type of tags to use")
parser.add_argument("--languages", type=str, default="ar,de,el,en,es,hi,ro,ru,th,tr,vi,zh", help="Comma-separated list of languages to process")
parser.add_argument("--do_scoring", type=bool, default=False, help="Whether to score contexts with COMET")
parser.add_argument("--scoring_threshold", type=float, default=0.0, help="Threshold for COMET scores")
parser.add_argument("--out_dir", type=str, default="./evaluation_data", help="Output directory for JSON files")
args = parser.parse_args()

EN_SOURCE_PATH = Path(args.en_source_path)
TEST_FOLDER_PATH = Path(args.test_folder_path)
MAX_LENGTH = args.max_length
MAX_LENGTH_TOKENIZED = args.max_length_tokenized
TAG_TYPE = args.tag_type if args.tag_type != "None" else None
LANGS = args.languages.split(",")
DO_SCORING = args.do_scoring
SCORING_THRESHOLD = args.scoring_threshold

OUT_DIR_SUFFIX = f"_{TAG_TYPE}" if TAG_TYPE else ""
OUT_PATH = Path(f"{args.out_dir}/XQUAD{OUT_DIR_SUFFIX}")

if DO_SCORING:
    from comet import download_model, load_from_checkpoint

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

with open(EN_SOURCE_PATH, "r") as f:
    master_data = json.load(f)

# make a qas id to paragraph id mapping
# generate paragraph id on the fly
qas_to_paragraph = {}
for i, item in enumerate(master_data['data']): 
    for j, paragraph in enumerate(item['paragraphs']):
        for qas in paragraph['qas']:
            qas_to_paragraph[qas['id']] = f"{i}-{j}"

paragraph_to_qas = defaultdict(list)
for qas_id, para_id in qas_to_paragraph.items():
    paragraph_to_qas[para_id].append(qas_id)

parallel_data = defaultdict(lambda: defaultdict(lambda: {"context": "", "qas": []}))
for testfile in TEST_FOLDER_PATH.glob("*.json"):
    _, src_lang = testfile.stem.split(".")
    
    with open(testfile, "r") as f:
        data = json.load(f)
    
    for item in data['data']:
        for paragraph in item['paragraphs']:
            for qas in paragraph['qas']:
                # if all the qas don't match the master data, skip (technically only need to check the first one, but just to be safe)
                paragraph_id = qas_to_paragraph[qas['id']]
                master_qas_ids = paragraph_to_qas[paragraph_id]
                if not set(master_qas_ids) == set(qas['id'] for qas in paragraph['qas']):
                    print(f"Warning: QAs in {paragraph_id} do not match master data, skipping.")
                    continue
                
                parallel_data_dict = parallel_data[paragraph_id]
                parallel_data_dict[src_lang]['context'] = paragraph['context']
                parallel_data_dict[src_lang]['qas'].append(qas)

print(f"Total paragraphs in parallel data: {len(parallel_data)}")

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
# for the time being, just have lg-en pairs
for src_lang in LANGS:
    if src_lang == "en":
        continue
    tgt_lang = "en"
    
    if DO_SCORING:
        # get comet scores for filtering
        comet_input = []
        para_id_list = []
        for id, paragraph in parallel_data.items():
            src_context = paragraph[src_lang]['context']
            tgt_context = paragraph[tgt_lang]['context']
            comet_input.append({"src": src_context, "mt": tgt_context})
            para_id_list.append(id)

        scores = model.predict(comet_input, batch_size=16,  gpus=1, accelerator="gpu").scores
        
        print(f"Comet scores for {src_lang}-{tgt_lang}: {sum(score >= 0.90 for score in scores)} out of {len(scores)} above 0.90")
        print(f"Comet scores for {src_lang}-{tgt_lang}: {sum(score >= 0.75 for score in scores)} out of {len(scores)} above 0.75")
        print(f"Comet scores for {src_lang}-{tgt_lang}: {sum(score >= 0.5 for score in scores)} out of {len(scores)} above 0.5")
        print(f"Comet scores for {src_lang}-{tgt_lang}: {sum(score < 0.25 for score in scores)} out of {len(scores)} below 0.25")
    
    tagged_data = []
    dup_tags = 0  # to count duplicate tags

    for id, paragraph in parallel_data.items():
        if src_lang not in paragraph or tgt_lang not in paragraph:
            continue
        src_context = paragraph[src_lang]['context']
        tgt_context = paragraph[tgt_lang]['context']
        
        if DO_SCORING:
            # filter out based on comet scores
            index = para_id_list.index(id)
            if scores[index] < SCORING_THRESHOLD:
                continue
        
        # if src_context_tagged or tgt_context_tagged is greater than max tokens, skip
        src_tokens = tokenizer(src_context, return_tensors='pt', truncation=False)
        tgt_tokens = tokenizer(tgt_context, return_tensors='pt', truncation=False)
        if src_tokens.input_ids.shape[1] > MAX_LENGTH or tgt_tokens.input_ids.shape[1] > MAX_LENGTH:
            continue
        
        # find the qas that have the same id in both languages
        common_qas = set(qas['id'] for qas in paragraph[src_lang]['qas']) & set(qas['id'] for qas in paragraph[tgt_lang]['qas'])
        if not common_qas:
            continue
        
        src_qas = {qas['id']: qas for qas in paragraph[src_lang]['qas'] if qas['id'] in common_qas}
        tgt_qas = {qas['id']: qas for qas in paragraph[tgt_lang]['qas'] if qas['id'] in common_qas}
        
        # Find the spans for tagging, regardless of tag type
        tag_num = 0
        offset_pairs = []  # to store all (start, end, tag) for inserting later
        for qas_id in common_qas:
            src_answers = src_qas[qas_id]['answers']
            tgt_answers = tgt_qas[qas_id]['answers']
            
            # check if there are any qas that have differing answer lengths (none in MLQA, but to be safe)
            if len(src_answers) != len(tgt_answers):
                raise ValueError(f"Different number of answers for qas {qas_id} in {src_lang} and {tgt_lang}")

            for src_answer, tgt_answer in zip(src_answers, tgt_answers):
                if TAG_TYPE == "squarebracket":
                    tag_open = '['
                    tag_close = ']'
                else:
                    if tag_num >= 26:
                        raise ValueError("Exceeded 26 unique tags (a-z). Extend logic if needed.")
                    tag_char = chr(ord('a') + tag_num)
                    tag_open = f"<{tag_char}>"
                    tag_close = f"</{tag_char}>"
                
                # get start and end indices
                src_start = src_answer['answer_start']
                src_end = src_start + len(src_answer['text'])

                tgt_start = tgt_answer['answer_start']
                tgt_end = tgt_start + len(tgt_answer['text'])
                
                # if the answer isn't present in the context, skip and warn
                # there are some weird inconsistencies in the dataset itself, nothing else to do
                # given the answer is in there, it's fairly safe to assume the context is correct
                if src_context[src_start:src_end] != src_answer['text']:
                    print(f"Warning: Answer text '{src_answer['text']}' not found in src context at {src_start}:{src_end} for qas {qas_id} in {src_lang}")
                    continue
                if tgt_context[tgt_start:tgt_end] != tgt_answer['text']:
                    print(f"Warning: Answer text '{tgt_answer['text']}' not found in tgt context at {tgt_start}:{tgt_end} for qas {qas_id} in {tgt_lang}")
                    continue
                
                # Check if the start/end pair already exists
            
                for existing in offset_pairs:
                    ln, start, end, tag_open_existing, tag_close_existing = existing
                    if ln == 'src' and start == src_start and end == src_end:
                        dup_tags += 1
                        break
                    if ln == 'tgt' and start == tgt_start and end == tgt_end:
                        dup_tags += 1
                        break
                else: # if no break occurred, add the tag
                    # Store with associated tag
                    offset_pairs.append(('src', src_start, src_end, tag_open, tag_close))
                    offset_pairs.append(('tgt', tgt_start, tgt_end, tag_open, tag_close))

                    tag_num += 1

        # if there are no tags to insert, just continue
        if not offset_pairs:
            print(f"No tags to insert for {src_lang}-{tgt_lang} with id {id}, skipping.")
            continue
        if TAG_TYPE:
            # Sort by start offset in reverse so insertions don't mess up positions
            for lang in ['src', 'tgt']:
                context = src_context if lang == 'src' else tgt_context
                
                # Build insertions with tie-break info: close tag (1) before open tag (0) at same index
                # for closes at same index, insert the one that starts the last
                insertions = []
                for l, start, end, tag_open, tag_close in offset_pairs:
                    if l == lang:
                        insertions.append((end, 1, -start, tag_close))  
                        insertions.append((start, 0, -end, tag_open))  

                # Sort descending
                insertions.sort(reverse=True)
                mod_context = context
                for point, _, _, tag in insertions:
                    context = context[:point] + tag + context[point:]

                if lang == 'src':
                    src_context_tagged = context
                else:
                    tgt_context_tagged = context  
        else:
            src_context_tagged = src_context
            tgt_context_tagged = tgt_context
        
        # MAX_LENGTH should be set with the extra tokens in mind, so check again just in case
        src_tokens_tagged = tokenizer(src_context_tagged, return_tensors='pt', truncation=False)
        tgt_tokens_tagged = tokenizer(tgt_context_tagged, return_tensors='pt', truncation=False)
        if src_tokens_tagged.input_ids.shape[1] > MAX_LENGTH_TOKENIZED or tgt_tokens_tagged.input_ids.shape[1] > MAX_LENGTH_TOKENIZED:
            raise ValueError(f"Tagged context for {src_lang} or {tgt_lang} exceeds max length after tagging. "
                             f"src: {src_tokens.input_ids.shape[1]}, tgt: {tgt_tokens.input_ids.shape[1]}, "
                             f"src_tagged: {src_tokens_tagged.input_ids.shape[1]}, tgt_tagged: {tgt_tokens_tagged.input_ids.shape[1]}",
                             f"src_context_tagged: {src_context_tagged}, tgt_context_tagged: {tgt_context_tagged}")
        
        tagged_data.append({
            "id": id,
            "src_context": src_context_tagged,
            "tgt_context": tgt_context_tagged,
            "qas": {
                "src": src_qas,
                "tgt": tgt_qas
            }
        })
    print(f"{dup_tags} duplicate tags skipped.")

            
    output_file = OUT_PATH / f"{src_lang}-{tgt_lang}" / f"test.{src_lang}-{tgt_lang}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing tagged data to {output_file}, total items: {len(tagged_data)}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in tagged_data:
            json.dump({
                "translation": {
                    src_lang: item["src_context"],
                    tgt_lang: item["tgt_context"]
                },
            }, f, ensure_ascii=False)
            f.write("\n")
            