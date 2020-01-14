# -*- coding: utf-8 -*-
# @Time    : 2020/1/13 11:30 上午
# @Author  : Jiangweiwei
# @mail    : zhongerqiandan@163.com

import numpy as np
import os
import json
from tokenization import FullTokenizer
import argparse
from tqdm import tqdm
from random import shuffle


# parser = argparse.ArgumentParser()
# parser.add_argument('--in_path', type = str, default =  "data")
# parser.add_argument('--out_path', type = str, default = "processed_data")
# vac_path = '/data/jiangweiwei/bertmodel/cased_L-12_H-768_A-12/vocab.txt'
# args = parser.parse_args()
# in_path = args.in_path
# out_path = args.out_path
# case_sensitive = False
#
# char_limit = 16
# # train_distant_file_name = os.path.join(in_path, 'train_distant.json')
# train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
# dev_file_name = os.path.join(in_path, 'dev.json')
# test_file_name = os.path.join(in_path, 'test.json')
#
# rel2id = json.load(open(os.path.join(in_path, 'rel2id.json'), "r"))
# id2rel = {v:u for u,v in rel2id.items()}
# json.dump(id2rel, open(os.path.join(in_path, 'id2rel.json'), "w"))
# # fact_in_train = set([])
# # fact_in_dev_train = set([])
#
# tokenizer = FullTokenizer(vac_path, do_lower_case=False)

class DocRedTokenizer:

    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def tokenize(self, orig_tokens):
        bert_tokens = []
        orig_to_tok_map = []
        for orig_token in orig_tokens:
            orig_to_tok_map.append(len(bert_tokens))
            bert_tokens.extend(tokenizer.tokenize(orig_token))
        bert_tokens.append("[SEP]")
        return bert_tokens, orig_to_tok_map

    def encode(self, bert_tokens):
        bert_input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        if len(bert_input_ids) > self.max_seq_len:
            bert_input_ids = bert_input_ids[:self.max_seq_len]
        else:
            bert_input_ids += [0] * (self.max_seq_len - len(bert_input_ids))
        segment_ids = [0] * self.max_seq_len
        return bert_input_ids, segment_ids


# doc_tokenizer = DocRedTokenizer(tokenizer, 512)

def build_data(data_file_name, rel2id, tokenizer, mode):
    ori_data = json.load(open(data_file_name))
    ori_data = ori_data[100:200]
    data = []

    for i in tqdm(range(len(ori_data))):
        Ls = [0]
        L = 0
        orig_tokens = []

        for x in ori_data[i]['sents']:
            L += len(x)
            Ls.append(L)
            orig_tokens += x
        bert_tokens, orig_to_tok_map = tokenizer.tokenize(orig_tokens)
        input_ids, segment_ids = tokenizer.encode(bert_tokens)
        vertexSet = ori_data[i]['vertexSet']

        for j in range(len(vertexSet)):
            for k in range(len(vertexSet[j])):
                vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                sent_id = vertexSet[j][k]['sent_id']
                dl = Ls[sent_id]
                pos1 = vertexSet[j][k]['pos'][0]
                pos2 = vertexSet[j][k]['pos'][1]
                vertexSet[j][k]['pos'] = (pos1 + dl, pos2 + dl)
                start = orig_to_tok_map[pos1 + dl]
                try:
                    end = orig_to_tok_map[pos2 + dl] - 1
                except:
                    end = len(bert_tokens) - 1
                vertexSet[j][k]['start'] = start
                vertexSet[j][k]['end'] = end

        if mode == 'train' or mode == 'dev' or mode == 'all':
            labels = ori_data[i].get('labels', [])
            for label in labels:
                rel = label['r']
                rel_id = rel2id[rel]
                h_id = label['h']
                t_id = label['t']
                for h_ver in vertexSet[h_id]:
                    if h_ver['end'] > tokenizer.max_seq_len - 1:
                        continue
                    for t_ver in vertexSet[t_id]:
                        if t_ver['end'] > tokenizer.max_seq_len - 1:
                            continue
                        item = {}
                        item['input_ids'] = input_ids
                        item['segment_ids'] = segment_ids
                        item['rel_id'] = rel_id
                        h_start, h_end = h_ver['start'], h_ver['end']
                        t_start, t_end = t_ver['start'], t_ver['end']
                        h_list, t_list = [], []
                        for pos in range(tokenizer.max_seq_len):
                            if pos >= h_start and pos <= h_end:
                                avg_num = 1 / (h_end - h_start + 1)
                                h_list.append(avg_num)
                            else:
                                h_list.append(0.)
                            if pos >= t_start and pos <= t_end:
                                avg_num = 1 / (t_end - t_start + 1)
                                t_list.append(avg_num)
                            else:
                                t_list.append(0.)
                        item['h_t_pos'] = [h_list, t_list]
                        data.append(item)

        else:
            for j in range(len(vertexSet) - 1):
                for k in range(j + 1, len(vertexSet)):
                    for first_ver in vertexSet[j]:
                        if first_ver['end'] > tokenizer.max_seq_len - 1:
                            continue
                        for second_ver in vertexSet[k]:
                            if second_ver['end'] > tokenizer.max_seq_len - 1:
                                continue
                            item = {'title': ori_data[i]['title']}
                            item['id'] = f'{i}-{j}-{k}'
                            item['h_idx'] = j
                            item['t_idx'] = k
                            # item['ver_id'] = {'first':j, 'second':k}
                            evidence = [first_ver['sent_id'], second_ver['sent_id']]
                            evidence = list(set(evidence))
                            evidence.sort()
                            item['evidence'] = evidence
                            item['input_ids'] = input_ids
                            item['segment_ids'] = segment_ids
                            first_start, first_end = first_ver['start'], first_ver['end']
                            second_start, second_end = second_ver['start'], second_ver['end']
                            first_list, second_list = [], []
                            for pos in range(tokenizer.max_seq_len):
                                if pos >= first_start and pos <= first_end:
                                    avg_num = 1 / (first_end - first_start + 1)
                                    first_list.append(avg_num)
                                else:
                                    first_list.append(0.)
                                if pos >= second_start and pos <= second_end:
                                    avg_num = 1 / (second_end - second_start + 1)
                                    second_list.append(avg_num)
                                else:
                                    second_list.append(0.)
                            item['h_t_pos'] = [first_list, second_list]
                            data.append(item)

                            item = {'title': ori_data[i]['title']}
                            item['id'] = f'{i}-{j}-{k}'
                            item['h_idx'] = k
                            item['t_idx'] = j

                            # item['ver_id'] = {'first':j, 'second':k}
                            evidence = [first_ver['sent_id'], second_ver['sent_id']]
                            evidence = list(set(evidence))
                            evidence.sort()
                            item['evidence'] = evidence
                            item['input_ids'] = input_ids
                            item['segment_ids'] = segment_ids
                            first_start, first_end = first_ver['start'], first_ver['end']
                            second_start, second_end = second_ver['start'], second_ver['end']
                            first_list, second_list = [], []
                            for pos in range(tokenizer.max_seq_len):
                                if pos >= first_start and pos <= first_end:
                                    avg_num = 1 / (first_end - first_start + 1)
                                    first_list.append(avg_num)
                                else:
                                    first_list.append(0.)
                                if pos >= second_start and pos <= second_end:
                                    avg_num = 1 / (second_end - second_start + 1)
                                    second_list.append(avg_num)
                                else:
                                    second_list.append(0.)
                            item['h_t_pos'] = [second_list, first_list]
                            data.append(item)
    if mode == 'train' or mode == 'dev' or mode == 'all':
        shuffle(data)
    return data

# train_data = build_data(train_annotated_file_name, rel2id, doc_tokenizer, 'train')
# json.dump(train_data, open(os.path.join(out_path, 'train.json'), 'w'))
# dev_data = build_data(dev_file_name, rel2id, doc_tokenizer, 'dev')
# json.dump(train_data, open(os.path.join(out_path, 'dev.json'), 'w'))
# test_data = build_data(test_file_name, rel2id, doc_tokenizer, 'test')
# json.dump(test_data, open(os.path.join(out_path, 'test.json'), 'w'))


# init(train_distant_file_name, rel2id, max_length = 512, is_training = True, suffix='')
# init(train_annotated_file_name, rel2id, max_length = 512, is_training = False, suffix='_train')
# init(dev_file_name, rel2id, max_length = 512, is_training = False, suffix='_dev')
# init(test_file_name, rel2id, max_length = 512, is_training = False, suffix='_test')
