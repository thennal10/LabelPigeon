import regex
import re

def insert_tags(context, qas_ls, tag_type="xml"):
    """
    Inserts tags around answers in the context based on the question-answer pairs.
    Args:
        context (str): The context in which to insert tags.
        qas_ls (list): List of question-answer pairs, each containing 'answers' with 'answer_start' and 'text'.
        tag_type (str): Type of tags to use, either "squarebracket" or "xml". Default is "xml".
    Returns:
        str: The context with tags inserted around the answers.
    """
    if tag_type not in ["squarebracket", "xml"]:
        raise ValueError("tag_type must be either 'squarebracket' or 'xml'.")
    
    tag_map = {} # map tags to their corresponding answer
    offset_pairs = []  # to store all (start, end, tag) for inserting later
    for qas in qas_ls:
        for ans_id, answer in enumerate(qas['answers']):
            if tag_type == "squarebracket":
                tag_open = '['
                tag_close = ']'
            else:
                if len(tag_map) >= 26:
                    raise ValueError("Exceeded 26 unique tags (a-z). Extend logic if needed.")
                tag_char = chr(ord('a') + len(tag_map))  # 'a' for first tag, 'b' for second, etc.
                tag_open = f"<{tag_char}>"
                tag_close = f"</{tag_char}>"
            
            # get start and end indices
            start = answer['answer_start']
            end = start + len(answer['text'])

            
            # if the answer isn't present in the context, skip and warn
            # if there are some weird inconsistencies in the dataset itself, nothing else to do
            # given the answer is in there, it's fairly safe to assume the context is correct
            if context[start:end] != answer['text']:
                print(f"Warning: Answer text '{answer['text']}' not found in src context at {start}:{end} for id {qas['id']}. Skipping.")
                continue
            
            # Check if the start/end pair already exists
            for existing in offset_pairs:
                existing_start, existing_end, _, _ = existing
                if start == existing_start and end == existing_end:
                    break
            else: # if no break occurred, add the tag
                # Store with associated tag
                offset_pairs.append((start, end, tag_open, tag_close))
                tag_map[tag_open] = (qas['id'], ans_id) # map tag to id and answer index

    # if there are no tags to insert, just continue
    if not offset_pairs:
        print(f"No tags to insert, skipping.")
        return context
    
    # Build insertions with tie-break info: close tag (1) before open tag (0) at same index
    # for closes at same index, do the one that starts the last
    insertions = []
    for start, end, tag_open, tag_close in offset_pairs:
        insertions.append((end, 1, -start, tag_close))  
        insertions.append((start, 0, -end, tag_open)) 

    # Sort descending
    insertions.sort(reverse=True)
    mod_context = context
    for point, _, _, tag in insertions:
        mod_context = mod_context[:point] + tag + mod_context[point:]

    return mod_context, tag_map

def insert_tags_from_spans(text, spans, tag_type="xml"):
    """
    Insert tags around given (start, end) spans (end exclusive).
    Args:
        text (str): Source text.
        spans (list[tuple[int,int]]): List of (start, end) character spans (end exclusive).
        tag_type (str): "xml" (default) or "squarebracket".
    Returns:
        (modified_text, tag_map)
          - modified_text: str with tags inserted
          - tag_map: dict mapping opening-tag -> span index
    """
    if tag_type not in ["squarebracket", "xml"]:
        raise ValueError("tag_type must be either 'squarebracket' or 'xml'.")

    tag_map = {}
    offset_pairs = []

    for i, (start, end) in enumerate(spans):
        # define tags
        if tag_type == "squarebracket":
            tag_open, tag_close = "[", "]"
        else:
            if len(offset_pairs) >= 26:
                # stop tagging here, need to extend logic
                print("Warning: Exceeded 26 unique tags (a-z). Stopping further tagging.")
                break
            tag_char = chr(ord('a') + len(offset_pairs))  # 'a', 'b', ...
            tag_open = f"<{tag_char}>"
            tag_close = f"</{tag_char}>"

        # avoid duplicate (start, end)
        for existing in offset_pairs:
            existing_start, existing_end, _, _ = existing
            if start == existing_start and end == existing_end:
                break
        else:
            offset_pairs.append((start, end, tag_open, tag_close))
            tag_map[tag_open] = i  # map to span index

    if not offset_pairs:
        return text, tag_map

    # Build insertions with tie-break info: close tag (1) before open tag (0) at same index
    # for closes at same index, do the one that starts the last
    insertions = []
    for start, end, tag_open, tag_close in offset_pairs:
        insertions.append((end,   1, -start, tag_close))
        insertions.append((start, 0, -end,   tag_open))

    insertions.sort(reverse=True)

    out = text
    for point, _, _, tag in insertions:
        out = out[:point] + tag + out[point:]

    return out, tag_map

def untag_text(text, tag_type="xml"):
    """
    Removes tags from the text and returns the cleaned text along with the tags found.
    Args:
        text (str): The input text containing tags.
        tag_type (str): Type of tags to remove, either "squarebracket" or "xml". Default is "xml".
    Returns:
        tuple: A tuple containing:
            - cleaned_text (str): The text with tags removed.
            - tags (list): A list of tuples where each tuple contains (tag_name, tag_text).
    """
    if tag_type not in ["squarebracket", "xml"]:
        raise ValueError("tag_type must be either 'squarebracket' or 'xml'.")

    tags = []
    if tag_type == "squarebracket":
        # WARNING: UNTESTED
        pattern = r"\[(?:[^\[\]]|(?R))*\]"
        matches = regex.findall(pattern, text, overlapped=True)
        remove_brackets = lambda x: x.replace('[', '').replace(']', '').strip()
        cleaned_text = remove_brackets(text)
        for m in matches:
            tags.append(('[]', remove_brackets(m)))
    else:  # xml, use BeautifulSoup 
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        cleaned_text = soup.get_text()
        for element in soup.find_all():
            tag_name = element.name
            tag_text = element.get_text()
            tags.append((tag_name, tag_text))

    return cleaned_text, tags


def has_token_stutter(text: str, max_allowed: int = 3) -> bool:
    """
    Detects if any token repeats consecutively more than max_allowed times.
    
    Args:
        text (str): Input string.
        max_allowed (int): Maximum allowed consecutive repetitions.
                          Default = 3 (so "the the the the" is flagged).
    
    Returns:
        bool: True if repetition detected, False otherwise.
    """
    # normalize whitespace
    tokens = re.sub(r"\s+", " ", text.strip()).split()
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            run += 1
            if run > max_allowed:
                return True
        else:
            run = 1
    return False