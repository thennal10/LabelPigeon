# Generate evaluation metrics for XML tag translation and overall translation quality using COMET. Uses outputs from run_evals.py.
# Tags are extracted from the reference directory structure ({dataset}_{tag_type}). For squarebrackets, we generate the tag translations
# using the provided MT model (e.g., NLLB) and then match them to the hypothesis tags using fuzzy matching, as in EasyProject.

import re
import sys
import json
import torch
import nltk
import jieba
import itertools
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from comet import download_model, load_from_checkpoint
from transformers import HfArgumentParser
from utils.utils import NLLB_CODE


@dataclass
class MetricsArguments:
    reference_dirs: str = field(
        default="evaluation_data/XQUAD_xml,evaluation_data/MLQA_xml",
        metadata={"help": "Comma-separated list of reference directories."}
    )
    hypothesis_dirs: str = field(
        default="outputs/xml/nllb-3b-lp",
        metadata={"help": "Comma-separated list of hypothesis directories."}
    )
    fuzzy_threshold: float = field(
        default=0.5,
        metadata={"help": "Fuzzy matching threshold for tag evaluation."}
    )
    mt_model_name: str = field(
        default=None,
        metadata={"help": "Machine translation model name for square bracket tag translation."}
    )
    alignment_model_name: str = field(
        default=None,
        metadata={"help": "Alignment model name for alignment-based tag extraction."}
    )

parser = HfArgumentParser(MetricsArguments)
if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
    # If we pass only one argument to the script and it's the path to a yaml file,
    # let's parse it to get our arguments.
    args = parser.parse_yaml_file(Path(sys.argv[1]))[0]
else:
    args = parser.parse_args_into_dataclasses()[0]

MT_TOKENIZER = None

if args.alignment_model_name:
    nltk.download('punkt_tab')

def fuzzy_match(a, b, threshold=0.9):
    """Return True if the string similarity between a and b is above the threshold."""
    return SequenceMatcher(None, a, b).ratio() >= threshold

def remove_tags(text):
    """Remove XML tags from a string."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def extract_tags_and_contents(tgt_text, src_text=None, tag_type="xml"):
    """Returns all (tag, inner_text) tuples recursively."""
    tags_and_contents = []
    if tag_type == "xml":
        soup = BeautifulSoup(tgt_text, "html.parser")
        # extract all tags and their inner text
        for tag in soup.find_all():
            tags_and_contents.append((tag.name, tag.get_text()))
    elif tag_type == "squarebracket":
        src_tags_and_contents = []
        # get the src tags
        soup = BeautifulSoup(src_text, "html.parser")
        for tag in soup.find_all():
            tl_text = MT_MODEL.generate(MT_TOKENIZER.encode(tag.get_text(), return_tensors='pt').cuda())
            tl_text = MT_TOKENIZER.decode(tl_text[0], skip_special_tokens=True)
            src_tags_and_contents.append((tag.name, tl_text))
        
        hyp_annotations = re.findall(r'\[(.*?)\]', tgt_text)
        # assign each hypothesis tag to the corresponding source tag using SequenceMatcher
        for hyp_ann in hyp_annotations:
            # just find the first match above 0.5
            for src_tag, src_content in src_tags_and_contents:
                if fuzzy_match(hyp_ann, src_content, 0.5):
                    tags_and_contents.append((src_tag, hyp_ann))
                    src_tags_and_contents.remove((src_tag, src_content))  # remove to avoid duplicates
                    break
    elif tag_type == "alignment":
        # Need to tokenize to words before aligning
        nltk_detokenizer = TreebankWordDetokenizer()
        
        def is_cjk(text):
            for ch in text:
                if '\u4e00' <= ch <= '\u9fff' or \
                '\u3040' <= ch <= '\u309f' or \
                '\u30a0' <= ch <= '\u30ff' or \
                '\u1100' <= ch <= '\u11ff' or \
                '\u3130' <= ch <= '\u318f' or \
                '\uac00' <= ch <= '\ud7af':
                    return True
            return False

        def pretokenize(text):
            if is_cjk(text):
                return list(jieba.cut(text))
            else:
                #return re.compile(r"\w+(?:'\w+)?|[^\w\s]", re.UNICODE).findall(text)
                return word_tokenize(text)
        
        def detokenize(tokens):
            if is_cjk("".join(tokens)):
                return "".join(tokens)
            else:
                return nltk_detokenizer.detokenize(tokens)
            
        # code from the awesome-align demo
        sent_src, sent_tgt = pretokenize(remove_tags(src_text)), pretokenize(tgt_text) # remember, src_text is xml_tagged
        token_src, token_tgt = [ALIGNMENT_TOKENIZER.tokenize(word) for word in sent_src], [ALIGNMENT_TOKENIZER.tokenize(word) for word in sent_tgt]
        wid_src, wid_tgt = [ALIGNMENT_TOKENIZER.convert_tokens_to_ids(x) for x in token_src], [ALIGNMENT_TOKENIZER.convert_tokens_to_ids(x) for x in token_tgt]
        ids_src, ids_tgt = ALIGNMENT_TOKENIZER.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=ALIGNMENT_TOKENIZER.model_max_length, truncation=True)['input_ids'], ALIGNMENT_TOKENIZER.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=ALIGNMENT_TOKENIZER.model_max_length)['input_ids']
        ids_src, ids_tgt = ids_src.cuda(), ids_tgt.cuda()
        sub2word_map_src = []
        for i, word_list in enumerate(token_src):
            sub2word_map_src += [i for x in word_list]
        sub2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_map_tgt += [i for x in word_list]

        # alignment
        align_layer = 8
        threshold = 1e-3
        with torch.no_grad():
            out_src = ALIGNMENT_MODEL(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
            out_tgt = ALIGNMENT_MODEL(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

            softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        align_words = set()
        for i, j in align_subwords:
            align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )

        # extract tags based on alignments
        soup = BeautifulSoup(src_text, "html.parser")
        for tag in soup.find_all():
            # if a continuous span of words is aligned to a continuous span in the target, extract that as a tag
            tag_tokens = pretokenize(tag.get_text())
            # find the start and end word indices of this tag in the source sentence
            tag_start, tag_end = None, None
            for i in range(len(sent_src) - len(tag_tokens) + 1):
                if sent_src[i:i+len(tag_tokens)] == tag_tokens:
                    tag_start, tag_end = i, i + len(tag_tokens) - 1
                    break
            if tag_start is None:
                # rare, usually indicates that the tokenization splits the tags or something
                # consider it a miss since alignment requires tokenization
                print(f"Warning: could not find tag content")
                continue
            
            aligned_tgt_indices = [j for (i, j) in align_words if tag_start <= i <= tag_end]
            if not aligned_tgt_indices:
                continue
            # check if this span is aligned to a continuous span in the target
            if aligned_tgt_indices == list(range(min(aligned_tgt_indices), max(aligned_tgt_indices)+1)):
                tag_content = detokenize(sent_tgt[min(aligned_tgt_indices):max(aligned_tgt_indices)+1])
                tags_and_contents.append((tag.name, tag_content))

    return tags_and_contents
    

def evaluate_tags(reference_texts, hypothesis_texts, source_texts, threshold=0.9, tag_type="xml"):
    total_ref_tags = 0 #p
    total_matches = 0 #tp
    total_mismatches = 0 #fp

    sentence_wise_matches = []
    for src, ref, hyp in tqdm(zip(source_texts, reference_texts, hypothesis_texts)):
        ref_tags = extract_tags_and_contents(ref) # reference is always xml
        hyp_tags = extract_tags_and_contents(hyp, src, tag_type)
        total_ref_tags += len(ref_tags)
        
        # find the corresponding tags in ref and hyp, and then see if they match
        matched = [False] * len(ref_tags)
        for i, (tag, ref_content) in enumerate(ref_tags):
            for h_tag, h_content in hyp_tags:
                if h_tag == tag: 
                    if fuzzy_match(ref_content, h_content, threshold):
                        total_matches += 1
                        matched[i] = True
                        hyp_tags.remove((h_tag, h_content))  # remove to avoid counting again
                        break
                    else:
                        total_mismatches += 1
                        break
        sentence_wise_matches.append({
            "source": src,
            "reference": ref,
            "hypothesis": hyp,
            "matches": matched,
        })

    precision = total_matches / (total_matches + total_mismatches) if (total_matches + total_mismatches) > 0 else 0
    recall = total_matches / total_ref_tags 
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1, sentence_wise_matches

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

if args.mt_model_name:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    # Load the machine translation model if provided
    print(f"Loading machine translation model from {args.mt_model_name}")
    MT_MODEL = AutoModelForSeq2SeqLM.from_pretrained(args.mt_model_name, device_map="auto")
    MT_MODEL.eval()
    if 'easyproject' in args.mt_model_name:
        MT_TOKENIZER = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", legacy_behaviour=True)
    else:
        MT_TOKENIZER = AutoTokenizer.from_pretrained(args.mt_model_name)

if args.alignment_model_name:
    from transformers import AutoModel, AutoTokenizer
    print(f"Loading alignment model from {args.alignment_model_name}")
    ALIGNMENT_MODEL = AutoModel.from_pretrained(args.alignment_model_name, device_map="auto")
    ALIGNMENT_MODEL.eval()
    ALIGNMENT_TOKENIZER = AutoTokenizer.from_pretrained(args.alignment_model_name)
for hypothesis_dir in args.hypothesis_dirs:
    if args.fuzzy_threshold != 0.9:
        eval_suffix = f"{args.fuzzy_threshold*10:02.0f}"
    else:
        eval_suffix = ""
    output_dir = Path(hypothesis_dir) / f"evals{eval_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for reference_dir in args.reference_dirs:
        if 'training_data' in reference_dir:
            evaluation_dataset = "test"
        else:
            evaluation_dataset = Path(reference_dir).name
        tag_type = evaluation_dataset.split('_')[1] if '_' in evaluation_dataset else None
        if tag_type == None and args.alignment_model_name: # set tag_type to alignment if we have an alignment model
            tag_type = "alignment"
        print(f"Evaluating {evaluation_dataset} in {reference_dir} against hypotheses in {hypothesis_dir}")

        # find all language pairs in the reference directory (all subdirectories)
        for lang_dir in Path(reference_dir).iterdir():
            if lang_dir.is_dir():
                first_lang, second_lang = lang_dir.name.split('-')
                lang_pairs = [lang_dir.name, f"{second_lang}-{first_lang}"]
                
                for lang_pair in lang_pairs:
                    source_lang, target_lang = lang_pair.split('-')

                    if tag_type == 'squarebracket': # we want the reference to be the xml tags so we actually have the order
                        ref_path = Path(reference_dir).parent / f"{evaluation_dataset.split('_')[0]}_xml" / lang_dir.name / f"test.{lang_dir.name}.json"
                        MT_TOKENIZER.src_lang, MT_TOKENIZER.tgt_lang = NLLB_CODE[source_lang], NLLB_CODE[target_lang]
                        MT_MODEL.config.forced_bos_token_id = MT_TOKENIZER.convert_tokens_to_ids(NLLB_CODE[target_lang])
                    elif tag_type == 'alignment': # for alignment based evaluation, we want the original reference as well
                        ref_path = Path(reference_dir).parent / f"{evaluation_dataset}_xml" / lang_dir.name / f"test.{lang_dir.name}.json"
                    else:
                        ref_path = lang_dir / f"test.{lang_dir.name}.json"
                    hyp_path = Path(hypothesis_dir) / "predictions" / f"{evaluation_dataset}.{lang_pair}.jsonl"
                    print(hyp_path)
                    # if it does not exist, skip
                    if not ref_path.is_file():
                        print(f"Reference file {ref_path} does not exist, skipping {lang_pair}.")
                        continue
                    if not hyp_path.is_file():
                        print(f"Hypothesis file {hyp_path} does not exist, skipping {lang_pair}.")
                        continue
                    
                    references = []
                    sources = []
                    with open(ref_path, 'r', encoding='utf-8') as ref_file:
                        for line in ref_file:
                            data = json.loads(line)
                            references.append(data['translation'][target_lang])
                            sources.append(data['translation'][source_lang])
                    
                    with open(hyp_path, 'r', encoding='utf-8') as hyp_file:
                        hypotheses = [line.strip() for line in hyp_file.readlines()]
                    
                    if len(references) != len(hypotheses):
                        raise ValueError(f"Mismatch in number of references and hypotheses for {lang_pair}")
                    
                    if tag_type == "alignment":
                        evaluation_output_name = f"{evaluation_dataset}_alignment"
                    else:
                        evaluation_output_name = evaluation_dataset
                    
                    # --- Tag evaluation ---
                    
                    if tag_type:
                        precision, recall, f1, tag_matches = evaluate_tags(references, hypotheses, sources, args.fuzzy_threshold, tag_type)
                        
                        # save tag matches to a file as json
                        with open(output_dir / f"{evaluation_output_name}.{lang_pair}.tag_matches", 'w', encoding='utf-8') as f:
                            json.dump(
                                {"f1": f1, "precision": precision, "recall": recall, "matches": tag_matches}, 
                                f, indent=4, ensure_ascii=False)
                        
                        print(f"[{lang_pair}] {tag_type.capitalize()} Tag Translation Evaluation:")
                        print(f"  Precision: {precision:.4f}")
                        print(f"  Recall:    {recall:.4f}")
                        print(f"  F1 Score:  {f1:.4f}")

                    # --- Comet evaluation --- 
                    
                    comet_data = [
                        {
                            "src": remove_tags(src),
                            "ref": remove_tags(ref),
                            "mt": remove_tags(hyp)
                        }
                        for src, ref, hyp in zip(sources, references, hypotheses)
                    ]
                    
                    scores = model.predict(comet_data, batch_size=32, gpus=1)
                    
                    print(f"[{lang_pair}] COMET Score: {scores.system_score:.4f}")
                    with open(output_dir / f"{evaluation_output_name}.{lang_pair}.comet", 'w', encoding='utf-8') as f:
                        f.write("reference\thypothesis\tscore\n")
                        for score, data in zip(scores.scores, comet_data):
                            ref = data['ref']
                            hyp = data['mt']
                            f.write(f"{ref}\t{hyp}\t{score}\n")
                        f.write(f"system_score: {scores.system_score}\n")