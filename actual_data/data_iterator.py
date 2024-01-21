import torch
import numpy as np
import torch.nn.functional as F
from data.data_utils import normalize_answer

IGNORE_INDEX = -100

class Config:
    max_document_size=384
    max_sequence_length=50
    
class HotpotDataset():
    
    def __init__(self, features, example_dict, graph_dict, batch_size, mode, entity_limit ):
        self.features = features
        self.batch_size=batch_size
        self.example_dict=example_dict
        self.graph_dict=graph_dict
        self.mode=mode
        self.entity_limit=entity_limit
        
        self.example_ptr = 0
        
        imp_features_list = [feature for feature in features if feature.qas_id in self.graph_dict]
        self.features = imp_features_list
        
    def __len__(self):
        return np.ceil(len(self.features)/len(self.batch_size))
    
    def pad_sequence(self, sequences, max_length, padding_value):
        return torch.stack([F.pad(seq, (0, max_length - seq.size(0)), value=padding_value) for seq in sequences])

    def collate_fn(self, batch):        
        context_idxs = torch.stack([item['context_idxs'] for item in batch])
        context_mask = torch.stack([item['context_mask'] for item in batch])
        
        tgt_idxs = torch.stack([item['tgt_idxs'] for item in batch])
        tgt_mask = torch.stack([item['tgt_mask'] for item in batch])
        tgt_lens = torch.stack([item['tgt_lens'] for item in batch])
        
        question_levels = torch.stack([item['question_level'] for item in batch])
        question_types = torch.stack([item['question_type'] for item in batch])
        
        entity_graphs = torch.stack([item['entity_graphs'] for item in batch])
        context_lens = torch.stack([item['context_lens'] for item in batch])
        
        ids = [item['ids'] for item in batch]
        context_batch_extend_vocab = torch.stack([item['context_batch_extend_vocab'] for item in batch])
        
        max_context_oovs = [item['max_context_oovs'] for item in batch]
        context_oovs = [item['context_oovs'] for item in batch]
        tgt_idxs_extend = torch.stack([item['tgt_idxs_extend'] for item in batch])
        
        entity_mapping = torch.stack([item['entity_mapping'] for item in batch])
        entity_lens = torch.stack([item['entity_lens'] for item in batch])
        entity_mask = torch.stack([item['entity_mask'] for item in batch])
        entity_label = torch.stack([item['entity_label'] for item in batch])
        start_mask = torch.stack([item['start_mask'] for item in batch])
        start_mask_weight = torch.stack([item['start_mask_weight'] for item in batch])
        
        bfs_mask = torch.stack([item['bfs_mask'] for item in batch])

        max_c_len = context_lens.max().item()
        max_t_len = tgt_lens.max().item()

        context_idxs = self.pad_sequence(context_idxs, max_c_len, padding_value=0)
        context_mask = self.pad_sequence(context_mask, max_c_len, padding_value=0)
        
        tgt_idxs = self.pad_sequence(tgt_idxs, max_t_len, padding_value=0)
        tgt_mask = self.pad_sequence(tgt_mask, max_t_len, padding_value=0)
        
        entity_graphs = self.pad_sequence(entity_graphs, self.entity_limit, padding_value=0)
        context_batch_extend_vocab = self.pad_sequence(context_batch_extend_vocab, max_c_len, padding_value=0)
        tgt_idxs_extend = self.pad_sequence(tgt_idxs_extend, max_t_len, padding_value=0)
        
        entity_mapping = self.pad_sequence(entity_mapping, self.entity_limit, padding_value=0)
        entity_lens = self.pad_sequence(entity_lens, self.entity_limit, padding_value=0)
        entity_mask = self.pad_sequence(entity_mask, self.entity_limit, padding_value=0)
        start_mask = self.pad_sequence(start_mask, self.entity_limit, padding_value=0)
        start_mask_weight = self.pad_sequence(start_mask_weight, self.entity_limit, padding_value=0)
        
        bfs_mask = self.pad_sequence(bfs_mask, self.entity_limit, padding_value=0)

        return {
            'context_idxs': context_idxs,
            'context_mask': context_mask,
            'tgt_idxs': tgt_idxs,
            'tgt_mask': tgt_mask,
            'tgt_lens': tgt_lens,
            'entity_graphs': entity_graphs,
            'context_lens': context_lens,
            'ids': ids,
            'context_batch_extend_vocab': context_batch_extend_vocab,
            'max_context_oovs': max_context_oovs,
            'context_oovs': context_oovs,
            'tgt_idxs_extend': tgt_idxs_extend,
            'entity_mapping': entity_mapping,
            'entity_lens': entity_lens,
            'entity_mask': entity_mask,
            'entity_label': entity_label,
            'start_mask': start_mask,
            'start_mask_weight': start_mask_weight,
            'bfs_mask': bfs_mask,
            
            'question_level': question_levels,
            'question_type': question_types,
        }
    
    def __getitem__(self, idx, config):
        start_id = idx * self.batch_size
        cur_batch_size = min(self.batch_size, len(self.features)-start_id)

        context_idxs = torch.LongTensor(self.batch_size, config.max_document_size)
        context_mask = torch.FloatTensor(self.batch_size, config.max_document_size)
        segment_idxs = torch.LongTensor(self.batch_size, config.max_document_size)

        # Graph and Mappings
        entity_graphs = torch.Tensor(self.batch_size, self.entity_limit, self.entity_limit)
        entity_mapping = torch.Tensor(self.batch_size, self.entity_limit, config.max_document_size)

        # pointer generator
        context_batch_extend_vocab = torch.LongTensor(self.bsz, config.max_document_size)
        tgt_idxs_extend = torch.LongTensor(self.bsz, config.max_seq_length)

        # Target tensor
        tgt_idxs = torch.LongTensor(self.bsz, config.max_seq_length)
        tgt_mask = torch.FloatTensor(self.bsz, config.max_seq_length)
        q_type = torch.LongTensor(self.bsz)


        start_mask = torch.FloatTensor(self.bsz, self.entity_limit)
        start_mask_weight = torch.FloatTensor(self.bsz, self.entity_limit)
        bfs_mask = torch.FloatTensor(self.bsz, self.n_layers, self.entity_limit)
        entity_label = torch.LongTensor(self.bsz)
        
        question_levels = [self.features[question]['question_level'] for question in cur_batch]
        question_types = [self.features[question]['question_type'] for question in cur_batch]

        while True:
            if self.example_ptr >= len(self.features):
                break
            start_id = self.example_ptr

            if self.mode == "decoding":                
                cur_batch_1 = self.features[start_id]
                cur_batch = [cur_batch_1 for _ in range(cur_batch_size)]
                self.example_ptr += 1

            else:
                cur_batch = self.features[start_id: start_id + cur_batch_size]
                cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)
                self.example_ptr += cur_batch_size


            ids = []
            max_sent_cnt = 0
            max_context_oovs = 0 
            max_entity_cnt = 0
            for mapping in entity_mapping:
                mapping.zero_()
            entity_label.fill_(IGNORE_INDEX)
           
            max_context_oovs = int(max([len(fe.doc_input_oovs) for fe in cur_batch]))
            context_oovs = []
            context_oovs = [case.doc_input_oovs for case in cur_batch]
            break_flag = False
            for i in range(len(cur_batch)):
                case = cur_batch[i]
                context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
                context_mask[i].copy_(torch.Tensor(case.doc_input_mask))

                tgt_idxs[i].copy_(torch.Tensor(case.tgt_question_ids))
                tgt_mask[i].copy_(torch.Tensor(case.tgt_question_mask))

                tgt_idxs_extend[i].copy_(torch.Tensor(case.tgt_question_extend_ids))

                context_batch_extend_vocab[i].copy_(torch.Tensor(case.doc_input_extend_vocab))

                tem_graph = self.graph_dict[case.qas_id]
                adj = torch.from_numpy(tem_graph['adj'])
                start_entities = torch.from_numpy(tem_graph['start_entities'])
                entity_graphs[i] = adj
                for l in range(self.n_layers):
                    bfs_mask[i][l].copy_(start_entities)
                    start_entities = bfs_step(start_entities, adj)

                start_mask[i].copy_(start_entities)
                start_mask_weight[i, :tem_graph['entity_length']] = start_entities.byte().any().float()
                
                ids.append(case.qas_id)
                answer = self.example_dict[case.qas_id].orig_answer_text
                for j, entity_span in enumerate(case.entity_spans[:self.entity_limit]):
                    _, _, ent, _  = entity_span
                    if normalize_answer(ent) == normalize_answer(answer):
                        entity_label[i] = j
                        break
                    
                entity_mapping[i] = torch.from_numpy(tem_graph['entity_mapping'])
                max_entity_cnt = max(max_entity_cnt, tem_graph['entity_length'])

            entity_lengths = (entity_mapping[:cur_batch_size] > 0).float().sum(dim=2)
            entity_lengths = torch.where((entity_lengths > 0), entity_lengths, torch.ones_like(entity_lengths))
            entity_mask = (entity_mapping > 0).any(2).float()

            input_lengths = (context_mask[:cur_batch_size] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            tgt_lengths = (tgt_mask[:cur_batch_size] > 0).long().sum(dim=1)
            max_t_len = int(tgt_lengths.max())

            yield {
                'context_idxs': context_idxs[:cur_batch_size, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_batch_size, :max_c_len].contiguous(),
                'tgt_idxs': tgt_idxs[:cur_batch_size, :max_t_len].contiguous(),  # the three are newly added, for the target part
                'tgt_mask': tgt_mask[:cur_batch_size, :max_t_len].contiguous(),
                'tgt_lens': tgt_lengths,
                'entity_graphs': entity_graphs[:cur_batch_size, :max_entity_cnt, :max_entity_cnt].contiguous(),
                'context_lens': input_lengths,
                'ids': ids,
                'context_batch_extend_vocab': context_batch_extend_vocab[:cur_batch_size, :max_c_len].contiguous(),
                'max_context_oovs': max_context_oovs,    
                'context_oovs': context_oovs,
                'tgt_idxs_extend' : tgt_idxs_extend[:cur_batch_size, :max_t_len].contiguous(),
                'entity_mapping': entity_mapping[:cur_batch_size, :max_entity_cnt, :max_c_len].contiguous(),
                'entity_lens': entity_lengths[:cur_batch_size, :max_entity_cnt].contiguous(),
                'entity_mask': entity_mask[:cur_batch_size, :max_entity_cnt].contiguous(),
                'entity_label': entity_label[:cur_batch_size].contiguous(),
                'start_mask': start_mask[:cur_batch_size, :max_entity_cnt].contiguous(),
                'start_mask_weight': start_mask_weight[:cur_batch_size, :max_entity_cnt].contiguous(),
                'bfs_mask': bfs_mask[:cur_batch_size, :, :max_entity_cnt].contiguous(),
                
                'question_level': question_levels,
                'question_type': question_types,
            }