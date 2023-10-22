import torch
import numpy as np
from torch.utils.data import Dataset
import os
import json
from collections import defaultdict
from .data_utils import combine_text, tokenize_text_bert, normalize_answer, no_of_topics
from torch.utils.data import DataLoader
import sys
import os
from transformers import BertTokenizer
# Add the root directory of your project to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# from preprocess_data.tokenization import BertTokenizer


class QGDataset(Dataset):
    
    def __init__(self, root_dir, max_seq_length=128, file_name="hotpot_train_v1.1.json", tokenizer = None, transform = None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.data = defaultdict(list) 
        self.max_seq_length = max_seq_length
        self.file_name = file_name
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        
        with open(os.path.join(self.root_dir, self.file_name), "r", encoding='utf-8') as f:
            for line in f:
                data_sample = json.loads(line)
                for row in range(len(data_sample)):
                    self.data['supporting_facts'].append(data_sample[row]['supporting_facts'])
                    self.data['level'].append(data_sample[row]['level'])
                    self.data['question'].append(data_sample[row]['question'])
                    self.data['context'].append([data_sample[row]['context']])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        context_full = []
        llm_token_count = 4096
        supporting_facts = self.data['supporting_facts'][index]
        question = self.data['question'][index]
        level = self.data['level'][index]
        no_topics = no_of_topics(self.data, 'context', index)
        
        for i in range(no_topics):
            context_full += self.data['context'][index][0][i][1]
          
        context_full = combine_text(context_full)  
        # tokenized_supporting_facts = tokenize_text_bert(supporting_facts)
        # tokenized_question = tokenize_text_bert(question)
        tokenized_level = tokenize_text_bert(level)
        tokenized_context_full = tokenize_text_bert(context_full)
        
        if len(tokenized_context_full) > llm_token_count:
            normalize_answer(context_full)
            
        sample = {
            # 'supporting_facts': supporting_facts,
            # 'question': tokenized_question,
            'level': tokenized_level,            
            'context': tokenized_context_full,
        }
        return sample
    
    def collate_fn(self, batch):
        input_ids = []
        attention_masks = []
        token_type_ids = []
        
        for sample in batch:
            tokenized_dict = self.tokenizer.encode_plus(
                sample['context'],
                sample['level'],
                add_special_tokens=True,
                max_length=self.max_seq_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids.append(tokenized_dict['input_ids'])
            attention_masks.append(tokenized_dict['attention_mask'])
            token_type_ids.append(tokenized_dict['token_type_ids'])

        sample =  {
            'input_ids': torch.stack(input_ids),
            'token_type_ids': torch.stack(token_type_ids),
            'attention_mask': torch.stack(attention_masks)
        }
        
        return sample
    
    def create_dataloader(self, dataset, batch_size=4, shuffle=True):
        """
        Creates a dataloader for the given dataset.
        """

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=dataset.collate_fn
        )

        return data_loader
        
    
dataset = QGDataset(root_dir="hotpot\\data", max_seq_length=128)
# print(dataset.__getitem__(0)['context'])
dataloader = dataset.create_dataloader(dataset, batch_size=4, shuffle=True)

# for batch in dataloader:
#     print(batch)