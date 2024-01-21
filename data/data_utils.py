from transformers import BertTokenizer
import re
import string
import collections
import numpy as np
import pickle
import spacy
import tqdm
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3

def tokenize_text_bert(text):
    tokens = tokenizer.tokenize(text)
    return tokens


def remove_stop_words_bert(tokens):
    stop_words = ["[CLS]", "[SEP]", "[PAD]"]  
    return [token for token in tokens if token not in stop_words]

# def remove_basic_stopwords(text):
#     stop_words = set(stopwords.words('english'))
#     words = nltk.word_tokenize(text)
#     filtered_words = [word for word in words if word.lower() not in stop_words]
#     return ' '.join(filtered_words)

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def replace_quotes(string):
    string.replace("''", '" ').replace("``", '" ')
    return string

def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
        if c == " ":
            continue
        ns_to_s_map[len(ns_chars)] = i
        ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

def no_of_topics(data, name, index):
    return len(data[name][index][0])

def combine_text(data):
    return ' '.join(data)
    
def encode_level(data):
    label_mapping = {'easy': 0, 'medium': 1, 'hard': 2}
    encoded_label = label_mapping[data['level']]
    return encoded_label

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def make_embedding(embedding_file, output_file, word2idx):
    word2embedding={}
    with open(embedding_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.split(" ")
        word = line[0]
        embedding = np.array(line[1:], dtype=np.float32)
        word2embedding[word] = embedding
    
    for word,idx in word2idx.items():
        if word in word2embedding:
            embedding[idx] = word2embedding[word]
        else:
            embedding[idx] = word2embedding[UNK_TOKEN]
    
    with open(output_file, 'w') as f:
        pickle.dump(embedding)
        
    return embedding

def find_token_spans(text, tokens):
    spans = []
    current_idx = 0

    for token in tokens:
        start_idx = text.find(token, current_idx)
        if start_idx == -1:
            print(f"Token '{token}' not found in the text.")
            raise ValueError("Token not found in text.")

        end_idx = start_idx + len(token)
        spans.append((start_idx, end_idx))

        current_idx = end_idx  

    return spans

def answer_span_sentences(context, span, answer_text, answer_end):
    nlp = spacy.load("en_core_web_lg")
    sentences = [sent.text for sent in nlp(context).sents]
    
    stop_idx=-1
    for idx,sentence in enumerate(sentences):
        if answer_text in sentence:
            chars = " ".join(sentences[:idx + 1])
            if len(chars) >= answer_end:
                stop_idx = idx
                break
    if stop_idx == -1:
        print(answer_text)
        print(context)
    truncated_sentences = sentences[:stop_idx + 1]
    truncated_context = " ".join(truncated_sentences).lower()
    return truncated_context

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)

    return spans


def convert_conll_format(examples, src_file, trg_file):
    src_fw = open(src_file, "w")
    trg_fw = open(trg_file, "w")
    for example in tqdm(examples):
        c_tokens = example["context_tokens"]
        if "\n" in c_tokens:
            print(c_tokens)
            print("new line")
        copied_tokens = deepcopy(c_tokens)
        q_tokens = example["ques_tokens"]
        # always select the first candidate answer
        start = example["y1s"][0]
        end = example["y2s"][0]

        for idx in range(start, end + 1):
            token = copied_tokens[idx]
            if idx == start:
                tag = "B_ans"
                copied_tokens[idx] = token + "\t" + tag
            else:
                tag = "I_ans"
                copied_tokens[idx] = token + "\t" + tag

        for token in copied_tokens:
            if "\t" in token:
                src_fw.write(token + "\n")
            else:
                src_fw.write(token + "\t" + "O" + "\n")

        src_fw.write("\n")
        question = " ".join(q_tokens)
        trg_fw.write(question + "\n")

    src_fw.close()
    trg_fw.close()
    
def make_hotpot_vocab(output_file, counter, max_vocab_size):
    sorted_vocab = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    word2idx = dict()
    word2idx[PAD_TOKEN] = 0
    word2idx[UNK_TOKEN] = 1
    word2idx[START_TOKEN] = 2
    word2idx[END_TOKEN] = 3

    for idx, (token, freq) in enumerate(sorted_vocab, start=4):
        if len(word2idx) == max_vocab_size:
            break
        word2idx[token] = idx
    with open(output_file, "wb") as f:
        pickle.dump(word2idx, f)

    return word2idx
            
def get_embedding(embedding_file, word2idx, output_file):
    word2embedding={}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.split(" ")
        word2idx[lines[0]] = np.array(lines[1:], dtype=np.float32)
    
    embedding = np.zeros((len(word2idx), 300), dtype=np.float32)
    for word, idx in word2idx.items():
        if word in word2embedding:
            embedding[idx] = word2embedding[word]
        else:
            embedding[idx] = word2embedding[UNK_TOKEN]
    
    with open(output_file, 'w') as f:
        pickle.dump(embedding,f)
    return embedding

def clean_entity(entity):
    Type = entity[1]
    Text = entity[0]
    if Type == "DATE" and ',' in Text:
        Text = Text.replace(' ,', ',')
    if '?' in Text:
        Text = Text.split('?')[0]
    Text = Text.replace("\'\'", "\"")
    Text = Text.replace("# ", "#")
    Text = Text.replace("''", '" ').replace("``", '" ').lower()
    return Text, Type        
            

