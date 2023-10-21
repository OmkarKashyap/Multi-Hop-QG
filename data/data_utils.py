from transformers import BertTokenizer
import re
import string
import collections

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


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


