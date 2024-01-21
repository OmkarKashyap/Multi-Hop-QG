import torch
from data.data_utils import convert_idx, replace_quotes, clean_entity
from preprocess_data.file_utils import load_json
from preprocess_data.tokenization import word_tokenize
import tqdm

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3

class HotpotExample:
    def __init__(self,
                 question_id,
                 question__type,
                 doc_tokens,
                 question_tokens,   
                 question_text,
                 para_start_end_position,
                 sent_start_end_position,
                 entity_start_end_position):
        
        self.question_id = question_id
        self.question__type = question__type
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.entity_start_end_position = entity_start_end_position
        
class InputFeaturesQG(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 para_spans,
                 sent_spans,
                 entity_spans,
                 tgt_question_tokens,    
                 tgt_question_ids,
                 tgt_question_mask,
                 tgt_question_extend_ids,
                 doc_input_extend_vocab,
                 doc_input_oovs,
                 
                 question_type=None,
                 question_level=None,
                 start_position=None,
                 end_position=None,
                 doc_length=None,
                 question_length=None,
                 answer_length=None,
                 named_entity_tags=None,
                ):
        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.para_spans = para_spans
        self.sent_spans = sent_spans
        self.entity_spans = entity_spans
        self.tgt_question_tokens = tgt_question_tokens
        self.tgt_question_ids = tgt_question_ids
        self.tgt_question_mask = tgt_question_mask
        self.tgt_question_extend_ids = tgt_question_extend_ids
        self.doc_input_extend_vocab = doc_input_extend_vocab
        self.doc_input_oovs = doc_input_oovs
        
        self.question_type = question_type
        self.question_level = question_level
        self.start_position = start_position
        self.end_position = end_position
        self.doc_length = doc_length
        self.question_length = question_length
        self.answer_length = answer_length
        self.named_entity_tags = named_entity_tags
        
class HotpotDataReader:
    def __init__(self, full_file, para_file, entity_file):
        self.full_file = full_file
        self.para_file = para_file
        self.entity_file = entity_file
        
    def read_hotpot_example_QG(self):
    
        full_data = load_json(self.full_file)
        para_data = load_json(self.para_file)
        entity_data = load_json(self.entity_file)
        examples = []
        
        for i,sample in tqdm(enumerate(full_data)):
            
            id = sample['_id']
            question_type = sample['type']
            question_level = sample['level'] 
            
            question = replace_quotes(sample['question']).lower()
            question_tokens = word_tokenize(question)    
                   
            doc_tokens = []
            sent_start_end_position = []
            para_start_end_position = []
            entity_start_end_position = []
            
            for paragraph in para_data[id]:
                title = paragraph[0]
                sents = paragraph[1]
                if title in entity_data[id]:
                    entities = entity_data[id][title]
                else:
                    entities = []
                    
            para_start_position = len(doc_tokens)
            
            for i, sent in enumerate(sents):
                sent = replace_quotes(sent).lower()
                sent_tokens = word_tokenize(sent)
                
                sent_start_word_id = len(doc_tokens)
                doc_tokens.extend(sent_tokens)
                sent_end_word_id = len(doc_tokens) - 1
                sent_start_end_position.append((sent_start_word_id, sent_end_word_id))
                
                sent_temp = " ".join(sent_tokens)
                sent_spans = convert_idx(sent_temp, sent_tokens)
                
                entity_pointer = 0
                for entity in entities:
                    entity_text, entity_type = clean_entity(entity)
                    
                    for entity in entities:
                        entity_text, entity_type = clean_entity(entity)
                        ent_tmp = " ".join(word_tokenize(entity_text))
                        entity_start = sent_temp.find(ent_tmp, find_start_index)
                        entity_span = []
                        if (entity_start != -1):
                            entity_end = entity_start + len(ent_tmp)
                            for idx, span in enumerate(sent_spans):
                                if not (entity_end <= span[0] or entity_start >= span[1]):
                                    entity_span.append(idx)
                            if len(entity_span) > 0:
                                ent_start_in_sent, ent_end_in_sent = entity_span[0], entity_span[-1]
                                ent_start_position = ent_start_in_sent + sent_start_word_id
                                ent_end_position = ent_end_in_sent + sent_start_word_id
                                entity_start_end_position.append((ent_start_position, ent_end_position, entity_text, entity_type))
                                entity_pointer += 1
                                find_start_index = entity_start + len(ent_tmp)
                        else:
                            break
                        
                    entities = entities[entity_pointer:]
                    
                    if len(doc_tokens) > config.max_document_size:
                        break
                                    
                para_end_position = len(doc_tokens) - 1
                para_start_end_position.append((para_start_position, para_end_position, title))
                
            context = " ".join(doc_tokens)
            token_spans = convert_idx(context, doc_tokens)
            
            example = HotpotExample(
                    question_id=id,
                    question_type=question_type,
                    question_level = question_level,
                    doc_tokens=doc_tokens,
                    question_text=question,
                    question_tokens=question_tokens,
                    para_start_end_position=para_start_end_position,
                    sent_start_end_position=sent_start_end_position,
                    entity_start_end_position=entity_start_end_position)
            examples.append(example)
        
        return examples
 
class HotpotFeatureConverter:
    def __init__(self, config):
        self.config = config

    def convert_examples_to_features(self, examples, word2idx, max_seq_length, max_query_length=50, max_answer_length=20):
        features = []

        for (example_index, example) in enumerate(tqdm(examples)):
            para_spans = []
            entity_spans = []
            sentence_spans = []
            all_doc_tokens = []
            tgt_question_tokens = []

            orig_to_tok_index = []
            orig_to_tok_back_index = []
            tok_to_orig_index = []

            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = [token]
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
                orig_to_tok_back_index.append(len(all_doc_tokens) - 1)

            if len(all_doc_tokens) > max_seq_length:
                all_doc_tokens = all_doc_tokens[:max_seq_length]

            tgt_question_tokens = example.question_tokens
            if len(tgt_question_tokens) > max_query_length - 2:
                tgt_question_tokens = tgt_question_tokens[:max_query_length - 2]

            doc_input_oovs = []
            tgt_question_extend_ids = []
            doc_input_extend_vocab = []

            doc_input_ids, doc_input_extend_vocab, doc_input_oovs = context2ids_hotpot(all_doc_tokens, word2idx,
                                                                                       max_seq_length)
            tgt_question_ids, tgt_question_extend_ids = question2ids_hotpot(tgt_question_tokens, word2idx,
                                                                            doc_input_oovs)

            for entity_span in example.entity_start_end_position:
                ent_start_position, ent_end_position = entity_span[0], entity_span[1]
                entity_spans.append((ent_start_position, ent_end_position, entity_span[2], entity_span[3]))

            for sent_span in example.sent_start_end_position:
                if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:
                    continue
                sent_start_position = orig_to_tok_index[sent_span[0]]
                sent_end_position = orig_to_tok_back_index[sent_span[1]]
                sentence_spans.append((sent_start_position, sent_end_position))

            for para_span in example.para_start_end_position:
                if para_span[0] >= len(orig_to_tok_index) or para_span[0] >= para_span[1]:
                    continue
                para_start_position = orig_to_tok_index[para_span[0]]
                para_end_position = orig_to_tok_back_index[para_span[1]]
                para_spans.append((para_start_position, para_end_position, para_span[2]))

            doc_input_mask = [1] * len(doc_input_ids)

            # Padding target question
            tgt_question_mask = [1] * len(tgt_question_ids)

            while len(doc_input_ids) < max_seq_length:
                doc_input_ids.append(0)
                doc_input_mask.append(0)
                doc_input_extend_vocab.append(0)

            while len(tgt_question_ids) < max_query_length:
                tgt_question_ids.append(0)
                tgt_question_mask.append(0)
                tgt_question_extend_ids.append(0)

            assert len(doc_input_ids) == max_seq_length
            assert len(doc_input_mask) == max_seq_length
            assert len(tgt_question_ids) == max_query_length
            assert len(tgt_question_mask) == max_query_length

            # Dropout out-of-bound span
            entity_spans = entity_spans[:self._largest_valid_index(entity_spans, max_seq_length)]
            sentence_spans = sentence_spans[:self._largest_valid_index(sentence_spans, max_seq_length)]

            features.append(
                InputFeaturesQG(question_id=example.question_id,
                                doc_tokens=all_doc_tokens,
                                doc_input_ids=doc_input_ids,
                                doc_input_mask=doc_input_mask,
                                para_spans=para_spans,
                                sent_spans=sentence_spans,
                                entity_spans=entity_spans,
                                tgt_question_tokens=tgt_question_tokens,
                                tgt_question_ids=tgt_question_ids,
                                tgt_question_mask=tgt_question_mask,
                                doc_input_oovs=doc_input_oovs,
                                tgt_question_extend_ids=tgt_question_extend_ids,
                                doc_input_extend_vocab=doc_input_extend_vocab,
                                question_type=example.question_type,
                                question_level=example.question_level,
                                )
            )

        return features

    def _largest_valid_index(self, spans, limit):
        for idx in range(len(spans)):
            if spans[idx][1] >= limit:
                return idx
                
def context2ids_hotpot(tokens, word2idx, max_seq_length):
    ids = list()
    extended_ids = list()
    oov_lst = list()

    for token in tokens:
        if token in word2idx:
            ids.append(word2idx[token])
            extended_ids.append(word2idx[token])
        else:
            ids.append(word2idx[UNK_TOKEN])
            if token not in oov_lst:
                oov_lst.append(token)
            extended_ids.append(len(word2idx) + oov_lst.index(token))
        if len(ids) == max_seq_length:
            break

    return ids, extended_ids, oov_lst

def question2ids_hotpot(tokens, word2idx, oov_lst):
    ids = list()
    extended_ids = list()
    ids.append(word2idx[START_TOKEN])
    extended_ids.append(word2idx[START_TOKEN])

    for token in tokens:
        if token in word2idx:
            ids.append(word2idx[token])
            extended_ids.append(word2idx[token])
        else:
            ids.append(word2idx[UNK_TOKEN])
            if token in oov_lst:
                extended_ids.append(len(word2idx) + oov_lst.index(token))
            else:
                extended_ids.append(word2idx[UNK_TOKEN])
    ids.append(word2idx[END_TOKEN])
    extended_ids.append(word2idx[END_TOKEN])

    return ids, extended_ids
