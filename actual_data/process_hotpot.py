import torch
import json
import tqdm
import nltk
import os
import spacy
nlp = spacy.load("en_core_web_lg")
from data.data_utils import make_hotpot_vocab, convert_conll_format, get_embedding, replace_quotes
from preprocess_data.tokenization import word_tokenize

embedding_file = "./glove/glove.840B.300d.txt"
embedding = "./hotpot/embedding.pkl"
src_word2idx_file = "./hotpot/word2idx.pkl"

train_hotpot = "./hotpot/data/hotpot_train_v1.1.json"
dev_hotpot = "./hotpot/data/hotpot_dev_distractor_v1.json"

train_src_file = "./hotpot-sent/para-train.txt"
train_trg_file = "./hotpot-sent/tgt-train.txt"
dev_src_file = "./hotpot-sent/para-dev.txt"
dev_trg_file = "./hotpot-sent/tgt-dev.txt"

test_src_file = "./hotpot-sent/para-test.txt"
test_trg_file = "./hotpot-sent/tgt-test.txt"
    
class DataProcessorHotpot:
    
    def __init__(self, file_path):
        self.file_path = file_path
    
    def process_data(self, file_name, mode, file_path=None):
        
        token_count = {}
        examples=[]
        self.file_path = os.path.join(self.file_path,file_name)
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for i,sample_data in tqdm(enumerate(data)):
            
            question = replace_quotes(sample_data['question']).lower()
            question_tokens = word_tokenize(question)
            for token in question_tokens:
                token_count[token] += 1
            
            question_type = sample_data['question_type']
            question_type_tokens = word_tokenize(question_type)
            
            question_level = sample_data['question_level']
            question_level_tokens = word_tokenize(question_level)
            
            fact=[]
            sent=[]
            supporting_facts = sample_data['supporting_facts']
            supporting_facts_tokens = word_tokenize(supporting_facts)
            for item in supporting_facts:
                fact.append(item[0])
                sent.append(item[1]) 
            
            context = sample_data['context']
            context_tokens = word_tokenize(context)
            for token in context_tokens:
                token_count[token] += 1
                
            contexts=[]
            context_tokens=[]
            for item in context:
                if item[0] in fact:
                    index = fact.index(item[0]) 
                    para = " ".join(item[1])
                    para = replace_quotes(para).lower()

                    sent = item[1][sent[index]]
                    sent = replace_quotes(sent).lower()

                    tokenized_para = word_tokenize(para)
                    tokenized_sent = word_tokenize(sent)

                    if mode == "para":
                        contexts.append(para)
                        context_tokens.extend(tokenized_para)
                    else:
                        contexts.append(sent)
                        context_tokens.extend(tokenized_sent)
                
            example = {'context_tokens':context_tokens, 'question_tokens':question_tokens, 'supporting_facts_tokens':supporting_facts_tokens, 'question_level_tokens':question_level_tokens, 'question_type_tokens':question_type_tokens}
            examples.append(example)
            
        return examples, token_count
    
    def make_dataset(self, mode, config):
        train_examples, token_count = self.process_data(train_hotpot, mode, config)
        convert_conll_format(train_examples, train_src_file, train_trg_file)
        word2idx = make_hotpot_vocab(src_word2idx_file, token_count, config.vocab_size)
        get_embedding(embedding_file, embedding, word2idx)

        dev_test_examples, _ = self.process_data(dev_hotpot, mode, config)
        num_dev = len(dev_test_examples) // 2
        dev_examples = dev_test_examples[:num_dev]
        test_examples = dev_test_examples[num_dev:]
        convert_conll_format(dev_examples, dev_src_file, dev_trg_file)
        convert_conll_format(test_examples, test_src_file, test_trg_file)

        
                

    

            