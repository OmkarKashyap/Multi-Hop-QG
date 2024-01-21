from process_hotpot import DataProcessorHotpot
from preprocess_data.file_utils import load_json
from data.data_utils import make_hotpot_vocab, convert_conll_format, get_embedding
from preprocess_data.tokenization import word_tokenize 
from sklearn.model_selection import train_test_split


embedding_file = "./glove/glove.840B.300d.txt"
embedding = "./hotpot/embedding.pkl"
src_word2idx_file = "./hotpot/word2idx.pkl"
train_hotpot = "./hotpot/data/hotpot_train_v1.1.json"
train_src_file = "./hotpot-sent/para-train.txt"
train_trg_file = "./hotpot-sent/tgt-train.txt"
dev_hotpot = "./hotpot/data/hotpot_dev_distractor_v1.json"
dev_src_file = "./hotpot-sent/para-dev.txt"
dev_trg_file = "./hotpot-sent/tgt-dev.txt"
test_src_file = "./hotpot-sent/para-test.txt"
test_trg_file = "./hotpot-sent/tgt-test.txt"

class Config:
    vocab_size = 10000  
    
def split_examples(examples, test_size=0.5, shuffle=True):
    if shuffle:
        return train_test_split(examples, test_size=test_size, shuffle=True)
    else:
        total_examples = len(examples)
        num_test = int(total_examples * test_size)
        return examples[num_test:], examples[:num_test]

def main():
    processor = DataProcessorHotpot(file_path='./hotpot')
    train_data = load_json(train_hotpot)

    #for train_data
    train_examples, token_count = processor.process_data(train_data, "para", Config())
    convert_conll_format(train_examples, train_src_file, train_trg_file)
    word2idx = make_hotpot_vocab(src_word2idx_file, token_count, Config.vocab_size)
    get_embedding(embedding_file, embedding, word2idx)
    
    #split data into train and dev sets
    dev_examples, _ = processor.process_data(dev_hotpot, "para", Config())    
    test_data, dev_data = split_examples(dev_examples, test_size=0.5, shuffle=True)
    
    convert_conll_format(dev_data, dev_src_file, dev_trg_file)
    convert_conll_format(test_data, test_src_file, test_trg_file)

if __name__ == "__main__":
    main()