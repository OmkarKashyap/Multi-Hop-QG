import os
import csv
import codecs
import yaml
import time
import numpy as np
import nltk
from nltk.translate import bleu_score
import pickle
import gzip
import sys
import torch
import torch.nn as nn
import random
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score

from config import set_config


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_datas(filename, trans_to_num=False):
    lines = open(filename, 'r').readlines()
    lines = list(map(lambda x: x.split(), lines))
    if trans_to_num:
        lines = [list(map(int, line)) for line in lines]
    return lines


def save_datas(data, filename, trans_to_str=False):
    if trans_to_str:
        data = [list(map(str, line)) for line in data]
    lines = list(map(lambda x: " ".join(x), data))
    with open(filename, 'w') as f:
        f.write("\n".join(lines))


def logging(file):
    def write_log(s):
        print(s)
        with open(file, 'a') as f:
            f.write(s)

    return write_log


def logging_csv(file):
    def write_csv(s):
        with open(file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(s)

    return write_csv


def format_time(t):
    return time.strftime("%Y-%m-%d-%H:%M:%S", t)


def eval_multi_bleu(references, candidate, log_path):
    ref_1, ref_2, ref_3, ref_4 = [], [], [], []
    candidate_new = []

    for cand in candidate:
        # ref_1.append(refs)
        cand_new = []
        for s in cand:
            if str(s) == "[SEP]" or str(s) == "[PAD]":
                break
            cand_new.append(s)
        candidate_new.append(cand_new)
        # if len(refs) > 1:
        #     ref_2.append(refs[1])
        # else:
        #     ref_2.append([])
        # if len(refs) > 2:
        #     ref_3.append(refs[2])
        # else:
        #     ref_3.append([])
        # if len(refs) > 3:
        #     ref_4.append(refs[3])
        # else:
        #     ref_4.append([])
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # log_path = log_path.strip('/')   #log_path ends with "/"
    ref_file_1 = log_path + 'reference_1.txt'
    # ref_file_2 = log_path + 'reference_2.txt'
    # ref_file_3 = log_path + 'reference_3.txt'
    # ref_file_4 = log_path + 'reference_4.txt'
    cand_file = log_path + 'candidate.txt'
    print(cand_file)
    print(ref_file_1)
    with open(ref_file_1, 'w') as f:
        for ref in references:
            f.write((" ".join(str(s) for s in ref) + '\n'))


    # with open(ref_file_2, 'w') as f:
    #     for s in ref_2:
    #         f.write(" ".join(s) + '\n')
    # with open(ref_file_3, 'w') as f:
    #     for s in ref_3:
    #         f.write(" ".join(s) + '\n')
    # with open(ref_file_4, 'w') as f:
    #     for s in ref_4:
    #         f.write(" ".join(s) + '\n')
    with open(cand_file, 'w') as f:
        for cand_new in candidate_new:
            f.write(" ".join(str(s) for s in cand_new) + '\n')
        
    temp = log_path + "result.txt"
    # command = "perl multi-bleu.perl " + ref_file_1 + " " + ref_file_2 + " " + ref_file_3 + " " + ref_file_4 + "<" + cand_file + "> " + temp
    command = "perl /home/ACL2020/GPG/multi-bleu.perl " + ref_file_1 + "<" + cand_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    try:
        print("====result is")
        print(result)
        # bleu = float(result.split(',')[0][7:])
        bleu_1 = float(result.split('/')[0][-4:])

    except ValueError:
        bleu_1 = 0
    return result, bleu_1


def eval_bleu(reference, candidate, log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # log_path = log_path.strip('/')
    ref_file = log_path + 'reference.txt'
    cand_file = log_path + 'candidate.txt'
    with codecs.open(ref_file, 'w') as f:
        for s in reference:
            f.write(" ".join(s) + '\n')
    with codecs.open(cand_file, 'w') as f:
        for s in candidate:
            f.write(" ".join(s).strip() + '\n')

    temp = log_path + "result.txt"
    command = "perl multi-bleu.perl " + ref_file + "<" + cand_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    try:
        bleu = float(result.split(',')[0][7:])
    except ValueError:
        bleu = 0
    return result, bleu


def write_result_to_file(candidates, references, log_path):
    assert len(references) == len(candidates), (len(references), len(candidates))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # log_path = log_path.strip('/')
    log_file = log_path + 'observe_result.tsv'
    with open(log_file, 'w') as f:
        for cand, ref in zip(candidates, references):
            f.write("".join(cand).strip() + '\t')
            f.write("".join(e.ori_title).strip() + '\t')
            # f.write(";".join(["".join(sent).strip() for sent in e.ori_content]) + '\t')
            f.write("".join(e.ori_original_content).strip() + '\t')
            # f.write("$$".join(["".join(comment).strip() for comment in e.ori_targets]) + '\t')
            f.write("\n")


def write_result_to_file_QG(source_qid, candidates, references, log_path):
    assert len(references) == len(candidates) == len(source_qid), (len(references), len(candidates), len(source_qid))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # log_path = log_path.strip('/')
    log_file = log_path + 'qid_observe_result.tsv'
    print(source_qid[0])
    print(candidates[0])
    print(references[0])
    with open(log_file, 'w') as f:
        for qid, cand, ref in zip(source_qid, candidates, references):
            cand_new = []
            for s in cand:
                if str(s) == "[SEP]" or str(s) == "[PAD]":
                    break
                # print(s)
                cand_new.append(s)
            # print((" ".join(str(s) for s in cand_new) + '\t'))
            # print((" ".join(str(s) for s in ref) + '\t'))
            f.write(qid + '\t')
            f.write(" ".join(str(s) for s in cand_new) + '\t')
            f.write(" ".join(str(s) for s in ref) + '\t')
            # # f.write(";".join(["".join(sent).strip() for sent in e.ori_content]) + '\t')
            # f.write("".join(e.ori_original_content).strip() + '\t')
            # f.write("$$".join(["".join(comment).strip() for comment in e.ori_targets]) + '\t')
            f.write("\n")


def write_qid_paragraphs_answers(qid, paragraphs, answers, log_path):
    assert len(qid) == len(paragraphs) == len(answers), (len(qid), len(paragraphs), len(answers))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # log_path = log_path.strip('/')
    log_file_qid_para = log_path + 'qid_para.tsv'
    log_file_qid_answer = log_path + 'qid_answer.tsv'
    with open(log_file_qid_para, 'w') as f:
        for q, p in zip(qid, paragraphs):
            f.write((str(q) + '\t'))
            f.write(" ".join(str(s) for s in p) + '\t')
            f.write("\n")
    with open(log_file_qid_answer, 'w') as f:
        for q, a in zip(qid, answers):
            f.write((str(q) + '\t'))
            f.write((str(a) + '\t'))
            f.write("\n")


def count_entity_num(candidates, tags):
    assert type(candidates) == list and type(tags) == list
    num = 0.
    for c, t in zip(candidates, tags):
        for word in c:
            if word in t:
                num += 1.
    return num / float(len(candidates))


def bow(word_list):
    word_dict = {}
    for word in word_list:
        if word not in word_dict:
            word_dict[word] = 1
    return word_dict

#--------------------------------------------------------------------------#
data_dir = config.set_config().data_dir

class Graph_utils():
    
    def __init__(self, config=None):
        
        self.data_dir = data_dir
        self.config = config
    
    def evaluate(self, tag, graphSage, classification):
        
        test_nodes = getattr(self.data_dir, tag+'_test')
        val_nodes = getattr(self.data_dir, tag+'_val')
        labels = getattr(self.data_dir, tag+'_labels')

        models = [graphSage, classification]

        params = []
        for model in models:
            for param in model.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    params.append(param)

        embs = graphSage(val_nodes)
        logists = classification(embs)
        _, predicts = torch.max(logists, 1)
        labels_val = labels[val_nodes]
        assert len(labels_val) == len(predicts)
        comps = zip(labels_val, predicts.data)

        vali_f1 = f1_score(labels_val, predicts.cpu().data, average="micro")
        print("Validation F1:", vali_f1)

        # if vali_f1 > max_vali_f1:
        # 	max_vali_f1 = vali_f1
        # 	embs = graphSage(test_nodes)
        # 	logists = classification(embs)
        # 	_, predicts = torch.max(logists, 1)
        # 	labels_test = labels[test_nodes]
        # 	assert len(labels_test) == len(predicts)
        # 	comps = zip(labels_test, predicts.data)

        # 	test_f1 = f1_score(labels_test, predicts.cpu().data, average="micro")
        # 	print("Test F1:", test_f1)

        # 	for param in params:
        # 		param.requires_grad = True

        # 	torch.save(models, 'models/model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, test_f1))

        # for param in params:
        # 	param.requires_grad = True

        # return max_vali_f1

    def get_gnn_embeddings(self, gnn_model,tag):
        print('Loading embeddings from trained GraphSAGE model.')
        features = np.zeros((len(getattr(self.data_dir, tag+'_labels')), gnn_model.out_size))
        nodes = np.arange(len(getattr(self.data_dir, tag+'_labels'))).tolist()
        b_sz = 500
        batches = math.ceil(len(nodes) / b_sz)
        embs = []
        for index in range(batches):
            nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
            embs_batch = gnn_model(nodes_batch)
            assert len(embs_batch) == len(nodes_batch)
            embs.append(embs_batch)
            # if ((index+1)*b_sz) % 10000 == 0:
            #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

        assert len(embs) == batches
        embs = torch.cat(embs, 0)
        assert len(embs) == len(nodes)
        print('Embeddings loaded.')
        return embs.detach()

    def train_classification(self,tag,graphSage, classification,epochs=800):
        print('Training Classification ...')
        c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
        # train classification, detached from the current graph
        #classification.init_params()
        b_sz = 50
        train_nodes = getattr(self.data_dir, tag+'_train')
        labels = getattr(self.data_dir, tag+'_labels')
        features = get_gnn_embeddings(graphSage, self.data_dir, tag)
        for epoch in range(epochs):
            train_nodes = shuffle(train_nodes)
            batches = math.ceil(len(train_nodes) / b_sz)
            visited_nodes = set()
            for index in range(batches):
                nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
                visited_nodes |= set(nodes_batch)
                labels_batch = labels[nodes_batch]
                embs_batch = features[nodes_batch]

                logists = classification(embs_batch)
                loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
                loss /= len(nodes_batch)
                # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

                loss.backward()
                
                nn.utils.clip_grad_norm_(classification.parameters(), 5)
                c_optimizer.step()
                c_optimizer.zero_grad()

            max_vali_f1 = self.evaluate(self,tag, graphSage, classification, epoch)
        return classification, max_vali_f1

    def apply_model(self,tag, graphSage, classification, unsupervised_loss, b_sz, unsup_loss,learn_method):
        test_nodes = getattr(self.data_dir, tag+'_test')
        val_nodes = getattr(self.data_dir, tag+'_val')
        train_nodes = getattr(self.data_dir, tag+'_train')
        labels = getattr(self.data_dir, tag+'_labels')

        if unsup_loss == 'margin':
            num_neg = 6
        elif unsup_loss == 'normal':
            num_neg = 100
        else:
            print("unsup_loss can be only 'margin' or 'normal'.")
            sys.exit(1)

        train_nodes = shuffle(train_nodes)

        models = [graphSage, classification]
        params = []
        for model in models:
            for param in model.parameters():
                if param.requires_grad:
                    params.append(param)

        optimizer = torch.optim.SGD(params, lr=0.1)
        optimizer.zero_grad()
        for model in models:
            model.zero_grad()

        batches = math.ceil(len(train_nodes) / b_sz)

        visited_nodes = set()
        for index in range(batches):
            nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]

            # extend nodes batch for unspervised learning
            # no conflicts with supervised learning
            nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
            visited_nodes |= set(nodes_batch)

            # get ground-truth for the nodes batch
            labels_batch = labels[nodes_batch]

            # feed nodes batch to the graphSAGE
            # returning the nodes embeddings
            embs_batch = graphSage(nodes_batch)

            if learn_method == 'sup':
                # superivsed learning
                logists = classification(embs_batch)
                loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
                loss_sup /= len(nodes_batch)
                loss = loss_sup
            elif learn_method == 'plus_unsup':
                # supervised learning
                logists = classification(embs_batch)
                loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
                loss_sup /= len(nodes_batch)
                # unsupervised learning
                if unsup_loss == 'margin':
                    loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
                elif unsup_loss == 'normal':
                    loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
                loss = loss_sup + loss_net
            else:
                if unsup_loss == 'margin':
                    loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
                elif unsup_loss == 'normal':
                    loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
                loss = loss_net

            print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
            loss.backward()
            for model in models:
                nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            optimizer.zero_grad()
            for model in models:
                model.zero_grad()

        return graphSage, classification
