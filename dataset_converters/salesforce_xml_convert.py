import json
import os
import random
import re
import string
from bs4 import BeautifulSoup
from html.parser import HTMLParser
from argparse import ArgumentParser


# get base directory from arguments
parser = ArgumentParser(description="Convert files to HF JSON format, optionally remapping tags.")
parser.add_argument("--input_dir", type=str, required=True, help="Base directory containing language subdirectories with source and target files")
parser.add_argument("--languages", type=str, required=False, help="Language pairs e.g., 'ende,enfr'", default="ende,enfr,enfi,enja,ennl,enru,enzh")
parser.add_argument("--out_dir", type=str, required=True, help="Output directory for JSON files")
parser.add_argument("--tag_type", type=str, default="alphabetic", choices=["alphabetic", "numeric"], help="Type of tags to use for remapping")
parser.add_argument("--remap", type=bool, default=True, help="Whether to remap tags from <a> to <z>")
parser.add_argument("--only_tags", type=bool, default=True, help="Whether to keep only the data with tags")
parser.add_argument("--remove_markers", type=bool, default=False, help="Whether to remove marker tags entirely")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

languages = args.languages.split(",")
random.seed(args.seed)

for lang in languages:
    base_dir = f"{args.input_dir}/{lang}"
    out_dir = args.out_dir
    src_lang, tgt_lang = lang[:2], lang[2:]

    os.makedirs(out_dir, exist_ok=True)

    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["text"]

    def write_split(split_name, pairs, out_dir):
        output_dir = os.path.join(out_dir, f"{tgt_lang}-{src_lang}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{split_name}.{tgt_lang}-{src_lang}.json")
        with open(output_path, "w", encoding="utf-8") as out_file:
            for tgt_text, src_text in pairs:
                json.dump({"translation": {tgt_lang: tgt_text, src_lang: src_text}}, out_file, ensure_ascii=False)
                out_file.write("\n")
        print(f"✅ Written {len(pairs)} examples to {output_path}")

    # Load English-German files
    train_src = load_json(os.path.join(base_dir, f"{lang}_{src_lang}_train.json"))
    train_tgt = load_json(os.path.join(base_dir, f"{lang}_{tgt_lang}_train.json"))
    test_src = load_json(os.path.join(base_dir, f"{lang}_{src_lang}_dev.json"))
    test_tgt = load_json(os.path.join(base_dir, f"{lang}_{tgt_lang}_dev.json"))

    # Verify matching keys
    def get_common_pairs(src_dict, tgt_dict):
        common_keys = set(src_dict.keys()) & set(tgt_dict.keys())
        if len(common_keys) != len(src_dict) or len(common_keys) != len(tgt_dict):
            print(f"⚠️ Warning: Only {len(common_keys)} / {len(src_dict)} (EN) and {len(tgt_dict)} (DE) keys matched")
        return [(tgt_dict[k], src_dict[k]) for k in sorted(common_keys)]

    # Prepare train/val split
    train_pairs = get_common_pairs(train_src, train_tgt)
    random.shuffle(train_pairs)
    split_idx = int(len(train_pairs) * 0.95)
    train_split = train_pairs[:split_idx]
    val_split = train_pairs[split_idx:]

    # Prepare test split
    test_pairs = get_common_pairs(test_src, test_tgt)

    if args.remap:
        class TagRenamer(HTMLParser):
            def __init__(self, tag_type='alphabetic'):
                super().__init__()
                self.tag_type = tag_type
                self.reset_state()
            
            def reset_state(self):
                self.output = []
                self.tag_map = {}
                if self.tag_type == 'alphabetic':
                    self.available_tags = list(string.ascii_lowercase)
                elif self.tag_type == 'numeric':
                    self.available_tags = [str(i) for i in range(1, 100)]
            
            def get_mapped_tag(self, tag):
                if tag not in self.tag_map:
                    if not self.available_tags:
                        raise ValueError("Ran out of replacement tags")
                    self.tag_map[tag] = self.available_tags.pop(0)
                return self.tag_map[tag]
            
            def handle_starttag(self, tag, attrs):
                new_tag = self.get_mapped_tag(tag)
                self.output.append(f"<{new_tag}>")
            
            def handle_endtag(self, tag):
                new_tag = self.get_mapped_tag(tag)
                self.output.append(f"</{new_tag}>")
            
            def handle_startendtag(self, tag, attrs):
                new_tag = self.get_mapped_tag(tag)
                self.output.append(f"<{new_tag}/>")
            
            def handle_data(self, data):
                self.output.append(data)
            
            def rename(self, text):
                self.reset_state()
                self.feed(text)
                return ''.join(self.output)

        def remap(pairs):
            # Process all sentence pairs
            renamer = TagRenamer(tag_type=args.tag_type)
            modified_pairs = []

            for de, en in pairs:
                renamer.reset_state()
                combined_text = de + ' ' + en
                renamer.feed(combined_text)
                tag_map_copy = dict(renamer.tag_map)  # Keep same mapping for both

                # Apply same tag map for DE and EN
                def remap_text(text, tag_map):
                    pattern = re.compile(r'</?[\w-]+/?>')
                    def replace_tag(match):
                        original_tag = match.group()
                        tag_name = re.sub(r'[</>]', '', original_tag)
                        mapped = tag_map.get(tag_name)
                        if original_tag.startswith('</'):
                            if args.tag_type == 'numeric':
                                return f"]{mapped}"
                            elif args.tag_type == 'alphabetic':
                                return f"</{mapped}>"
                        elif original_tag.endswith('/>'):
                            if args.tag_type == 'numeric':
                                return f"]{mapped}"
                            elif args.tag_type == 'alphabetic':
                                return f"<{mapped}/>"
                        else:
                            if args.tag_type == 'numeric':
                                return f"{mapped}["
                            elif args.tag_type == 'alphabetic':
                                return f"<{mapped}>"
                    return pattern.sub(replace_tag, text)

                # check if there are any tags in the text
                if renamer.tag_map:
                    en_remapped = remap_text(en, tag_map_copy)
                    de_remapped = remap_text(de, tag_map_copy)
                    modified_pairs.append((de_remapped, en_remapped))
                else:
                    # skip if only_tags, else append original texts
                    if not args.only_tags:
                        modified_pairs.append((de, en))
            return modified_pairs

        # Remap tags in train, val, and test splits
        train_split = remap(train_split)
        val_split = remap(val_split)
        test_pairs = remap(test_pairs)

    if args.remove_markers:
        def remove_tags(text):
            """Remove XML tags from a string."""
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text()
        def remove_tags_from_pairs(pairs):
            return [(remove_tags(tgt), remove_tags(src)) for tgt, src in pairs]
        train_split = remove_tags_from_pairs(train_split)
        val_split = remove_tags_from_pairs(val_split)
        test_pairs = remove_tags_from_pairs(test_pairs)

    # Write output files
    write_split("train", train_split, out_dir)
    write_split("valid", val_split, out_dir)
    write_split("test", test_pairs, out_dir)
