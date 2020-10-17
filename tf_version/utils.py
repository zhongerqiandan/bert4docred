import os
import json
import random
from collections import defaultdict

import sklearn.metrics
import numpy as np
from bert import tokenization
from tqdm import tqdm

bert_model_dir = "/data/jiangweiwei/bertmodel/uncased_L-12_H-768_A-12"
vocab_file = os.path.join(bert_model_dir, "vocab.txt")
tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

with open("../data/ner2id.json", 'r') as jf:
    ner2id = json.load(jf)
with open("../data/rel2id.json", 'r') as jf:
    rel2id = json.load(jf)
    id2rel = {rel2id[r]: r for r in rel2id}

train_relation_fact = set()


def convert_single_example(example):
    vertex_set = example["vertexSet"]
    sentences = example["sents"]
    mentions_in_sentence = get_mentions_in_each_sentence(example)
    for i, sentence in enumerate(sentences):
        sentences[i], vertex_set = convert_single_sentence(
            sentence, vertex_set, mentions_in_sentence[i])
    return vertex_set, sentences


def convert_single_sentence(sentence, vertex_set, mentions):
    # Token map will be an int -> int mapping between the `orig_tokens` index and the `tokens` index.
    token_map = []
    tokens = ["[CLS]"]
    for orig_token in sentence:
        token_map.append(len(tokens))
        tokens.extend(tokenizer.tokenize(orig_token))
    # Insert entity markers in tokens
    count = 0
    for mention in mentions:
        [i, j] = mention
        s = token_map[vertex_set[i][j]['pos'][0]]
        if vertex_set[i][j]['pos'][1] == len(token_map):
            e = len(tokens)
        else:
            e = token_map[vertex_set[i][j]['pos'][1]]
        vertex_set[i][j]['pos'] = [s + count * 2, e + count * 2 + 1]
        tokens.insert(s + count * 2, '[unused0]')
        tokens.insert(e + count * 2 + 1, '[unused1]')
        count += 1
    return tokens, vertex_set


def get_mentions_in_each_sentence(example):
    vertex_set = example["vertexSet"]
    sentences = example["sents"]
    num_sentences = len(sentences)
    mentions_in_sentence = {i: [] for i in range(num_sentences)}
    for i, vertex in enumerate(vertex_set):
        for j, mention in enumerate(vertex):
            sent_id = mention['sent_id']
            mentions_in_sentence[sent_id].append([i, j])
    for i in mentions_in_sentence:
        mentions_in_sentence[i].sort(key=lambda x: vertex_set[x[0]][x[1]]['pos'][0])
    return mentions_in_sentence


def get_data_property(file_list):
    max_entity_num = 0
    max_sentence_num = 0
    for file in file_list:
        with open(file, 'r') as jf:
            examples = json.load(jf)
        for example in examples:
            max_entity_num = max(len(example["vertexSet"]), max_entity_num)
            max_sentence_num = max(len(example["sents"]), max_sentence_num)
    max_relation_num = max_entity_num * (max_entity_num - 1)
    return max_entity_num, max_sentence_num, max_relation_num


def load_data(file_name, max_entity_num, max_sentence_num, max_relation_num,
              max_seq_length=512, is_train=True):
    with open(file_name, 'r') as jf:
        examples = json.load(jf)

    example_num = len(examples)
    max_node_num = max_entity_num + max_sentence_num

    len_vertex = []
    sent_num = []
    tokens = []
    input_ids = []
    segment_ids = []
    seq_masks = []
    entity_types = np.zeros((example_num, max_node_num), dtype=np.int32)
    entity_masks = np.zeros((example_num, max_entity_num, max_seq_length), dtype=np.float32)
    sentence_masks = np.zeros((example_num, max_sentence_num, max_seq_length), dtype=np.int32)
    attention_masks = np.zeros((example_num, max_node_num, max_node_num), dtype=np.int32)
    entity_attention_masks = np.zeros((example_num, max_node_num, max_node_num), dtype=np.int32)
    relation_masks = np.zeros((example_num, max_relation_num), dtype=np.int32)
    head_masks = np.zeros((example_num, max_relation_num, max_entity_num), dtype=np.int32)
    tail_masks = np.zeros((example_num, max_relation_num, max_entity_num), dtype=np.int32)
    multi_labels = np.zeros((example_num, max_relation_num, len(rel2id)), dtype=np.int32)
    # evidence = np.zeros((example_num, max_relation_num, len(rel2id), max_sentence_num), dtype=np.int32)
    # evidence_mask = np.zeros((example_num, max_relation_num, len(rel2id)), dtype=np.int32)
    not_na_labels = []
    is_in_train = []
    for idx, example in enumerate(tqdm(examples)):
        sent_num.append(len(example['sents']))
        vertex_set, sentences = convert_single_example(example)
        len_vertex.append(len(vertex_set))
        segment_id = []
        text = []
        cur_seq_length = 0
        sentence_set = []
        for i, sentence in enumerate(sentences):
            text.extend(sentence)
            segment_id.extend([i + 1] * len(sentence))
            if cur_seq_length < max_seq_length:
                sentence_set.append(cur_seq_length)
                sentence_masks[idx, i, cur_seq_length] = 1
            cur_seq_length += len(sentence)
        if cur_seq_length > max_seq_length:
            text = text[:max_seq_length]
            segment_id = segment_id[:max_seq_length]
            seq_mask = [1] * max_seq_length
        else:
            text.extend(["[PAD]"] * (max_seq_length - cur_seq_length))
            segment_id.extend([0] * (max_seq_length - cur_seq_length))
            seq_mask = [1] * cur_seq_length + [0] * (max_seq_length - cur_seq_length)
        tokens.append(text)
        input_id = tokenizer.convert_tokens_to_ids(text)
        input_ids.append(input_id)
        segment_ids.append(segment_id)
        seq_masks.append(seq_mask)

        relation_pairs = defaultdict(list)
        pair_evidence = defaultdict(list)
        labels = example.get("labels", [])
        for label in labels:
            relation_pairs[(label['h'], label['t'])].append(rel2id[label['r']])
            pair_evidence[(label['h'], label['t'])].append((rel2id[label['r']], label['evidence']))
        pair_counter = 0
        not_na_label = []
        in_train = set()
        for h, e1 in enumerate(vertex_set):
            for mention in e1:
                if mention['sent_id'] < len(sentence_set):
                    mention_pos = sentence_set[mention['sent_id']] + mention['pos'][0]
                    if mention_pos < max_seq_length:
                        entity_types[idx, h] = ner2id[mention['type']]
                        entity_masks[idx, h, mention_pos] = 1.0 / len(e1)
            for t, e2 in enumerate(vertex_set):
                if h != t:
                    head_masks[idx, pair_counter, h] = 1
                    tail_masks[idx, pair_counter, t] = 1
                    for r in relation_pairs[(h, t)]:
                        multi_labels[idx, pair_counter, r] = 1
                        not_na_label.append((h, t, r))
                        if is_train:
                            for m1 in e1:
                                for m2 in e2:
                                    train_relation_fact.add((m1["name"], m2["name"], r))
                        else:
                            for m1 in e1:
                                for m2 in e2:
                                    if (m1["name"], m2["name"], r) in train_relation_fact:
                                        in_train.add((h, t, r))
                    if not relation_pairs[(h, t)]:
                        multi_labels[idx, pair_counter, 0] = 1
                    # for n in pair_evidence[(h, t)]:
                    #     evidence[idx, pair_counter, n[0], n[1]] = 1
                    #     evidence_mask[idx, pair_counter, n[0]] = 1
                    pair_counter += 1
        not_na_labels.append(not_na_label)
        is_in_train.append(in_train)
        relation_masks[idx, : pair_counter] = 1
        attention_array = np.zeros(max_node_num, dtype=np.int32)
        attention_array[: len(vertex_set)] = 1
        attention_array[max_entity_num: max_entity_num + len(sentence_set)] = 1
        attention_masks[idx, : len(vertex_set)] = attention_array
        attention_masks[idx, max_entity_num: max_entity_num + len(sentence_set)] = attention_array
        entity_attention_masks[idx, : len(vertex_set)] = attention_array
        entity_attention_masks[idx, max_entity_num: max_entity_num + len(sentence_set)] = attention_array
        entity_attention_masks[idx, :max_entity_num , max_entity_num:] = 0

    dataset = {
        "tokens": tokens,
        "input_ids": np.asarray(input_ids),
        "segment_ids": np.asarray(segment_ids),
        "input_mask": np.asarray(seq_masks),
        "len_vertex": len_vertex,
        "entity_types": entity_types,
        "entity_mask": entity_masks,
        "sentence_mask": sentence_masks,
        "attention_mask": attention_masks,
        "entity_attention_mask": entity_attention_masks,
        "relation_mask": relation_masks,
        "head_mask": head_masks,
        "tail_mask": tail_masks,
        "multi_labels": multi_labels,
        "not_na_labels": not_na_labels,
        "is_in_train": is_in_train,
        "sent_num": sent_num
    }
    return dataset





# def load_data(file_name, max_entity_num, max_sentence_num, max_relation_num,
#               max_seq_length=512, is_train=True):
#     with open(file_name, 'r') as jf:
#         examples = json.load(jf)
#     example_num = len(examples)
#     sent_num = []
#     len_vertex = []
#     tokens = []
#     input_ids = []
#     segment_ids = []
#     seq_masks = []
#     entity_masks = np.zeros((example_num, max_entity_num, max_seq_length), dtype=np.float32)
#     sentence_masks = np.zeros((example_num, max_sentence_num, max_seq_length), dtype=np.int32)
#     max_node_num = max_entity_num + max_sentence_num
#     attention_masks = np.zeros((example_num, max_node_num, max_node_num), dtype=np.int32)
#     relation_masks = np.zeros((example_num, max_relation_num), dtype=np.int32)
#     head_masks = np.zeros((example_num, max_relation_num, max_entity_num), dtype=np.int32)
#     tail_masks = np.zeros((example_num, max_relation_num, max_entity_num), dtype=np.int32)
#     multi_labels = np.zeros((example_num, max_relation_num, len(rel2id)), dtype=np.int32)
#     evidence = np.zeros((example_num, max_relation_num, len(rel2id), max_sentence_num), dtype=np.int32)
#     evidence_mask = np.zeros((example_num, max_relation_num, len(rel2id)), dtype=np.int32)
#     not_na_labels = []
#     is_in_train = []
#     for idx, example in enumerate(tqdm(examples)):
#         sent_num.append(len(example['sents']))
#         vertex_set, sentences = convert_single_example(example)
#         len_vertex.append(len(vertex_set))
#         segment_id = []
#         text = []
#         cur_seq_length = 0
#         len_sentence = []
#         for i, sentence in enumerate(sentences):
#             text.extend(sentence)
#             segment_id.extend([i + 1] * len(sentence))
#             if cur_seq_length < max_seq_length:
#                 len_sentence.append(cur_seq_length)
#                 sentence_masks[idx, i, cur_seq_length] = 1
#             cur_seq_length += len(sentence)
#         if cur_seq_length > max_seq_length:
#             text = text[:max_seq_length]
#             segment_id = segment_id[:max_seq_length]
#             seq_mask = [1] * max_seq_length
#         else:
#             text.extend(["[PAD]"] * (max_seq_length - cur_seq_length))
#             segment_id.extend([0] * (max_seq_length - cur_seq_length))
#             seq_mask = [1] * cur_seq_length + [0] * (max_seq_length - cur_seq_length)
#         tokens.append(text)
#         input_id = tokenizer.convert_tokens_to_ids(text)
#         input_ids.append(input_id)
#         segment_ids.append(segment_id)
#         seq_masks.append(seq_mask)
#
#         relation_pairs = defaultdict(list)
#         pair_evidence = defaultdict(list)
#         labels = example.get("labels", [])
#         for label in labels:
#             relation_pairs[(label['h'], label['t'])].append(rel2id[label['r']])
#             pair_evidence[(label['h'], label['t'])].append((rel2id[label['r']], label['evidence']))
#         pair_counter = 0
#         not_na_label = []
#         in_train = set()
#         for h, e1 in enumerate(vertex_set):
#             for mention in e1:
#                 if mention['sent_id'] < len(len_sentence):
#                     mention_pos = len_sentence[mention['sent_id']] + mention['pos'][0]
#                     if mention_pos < max_seq_length:
#                         entity_masks[idx, h, mention_pos] = 1.0 / len(e1)
#             for t, e2 in enumerate(vertex_set):
#                 if h != t:
#                     head_masks[idx, pair_counter, h] = 1
#                     tail_masks[idx, pair_counter, t] = 1
#                     for r in relation_pairs[(h, t)]:
#                         multi_labels[idx, pair_counter, r] = 1
#                         not_na_label.append((h, t, r))
#                         if is_train:
#                             for m1 in e1:
#                                 for m2 in e2:
#                                     train_relation_fact.add((m1["name"], m2["name"], r))
#                         else:
#                             for m1 in e1:
#                                 for m2 in e2:
#                                     if (m1["name"], m2["name"], r) in train_relation_fact:
#                                         in_train.add((h, t, r))
#                     if not relation_pairs[(h, t)]:
#                         multi_labels[idx, pair_counter, 0] = 1
#                     for n in pair_evidence[(h, t)]:
#                         evidence[idx, pair_counter, n[0], n[1]] = 1
#                         evidence_mask[idx, pair_counter, n[0]] = 1
#                     pair_counter += 1
#         not_na_labels.append(not_na_label)
#         is_in_train.append(in_train)
#         relation_masks[idx, : pair_counter] = 1
#         attention_array = np.zeros(max_node_num, dtype=np.int32)
#         attention_array[: len(sentences)] = 1
#         attention_array[max_sentence_num: max_sentence_num + len(vertex_set)] = 1
#         attention_masks[idx, : len(sentences)] = attention_array
#         attention_masks[idx, max_sentence_num: max_sentence_num + len(vertex_set)] = attention_array
#
#     dataset = {
#         "input_ids": np.asarray(input_ids),
#         "segment_ids": np.asarray(segment_ids),
#         "input_mask": np.asarray(seq_masks),
#         "len_vertex": len_vertex,
#         "entity_mask": entity_masks,
#         "sentence_mask": sentence_masks,
#         "attention_mask": attention_masks,
#         "relation_mask": relation_masks,
#         "head_mask": head_masks,
#         "tail_mask": tail_masks,
#         "multi_labels": multi_labels,
#         "evidence": evidence,
#         "evidence_mask": evidence_mask,
#         "not_na_labels": not_na_labels,
#         "is_in_train": is_in_train,
#         "sent_num": sent_num}
#     return dataset


def load_test_data(file_name, max_entity_num, max_sentence_num,
                   max_relation_num, max_seq_length=512):
    with open(file_name, 'r') as jf:
        examples = json.load(jf)[:5]
    example_num = len(examples)
    max_node_num = max_entity_num + max_sentence_num

    sent_num = []
    titles = []
    len_vertex = []
    tokens = []
    input_ids = []
    segment_ids = []
    seq_masks = []
    entity_masks = np.zeros((example_num, max_entity_num, max_seq_length), dtype=np.float32)
    sentence_masks = np.zeros((example_num, max_sentence_num, max_seq_length), dtype=np.int32)
    attention_masks = np.zeros((example_num, max_node_num, max_node_num), dtype=np.int32)
    relation_masks = np.zeros((example_num, max_relation_num), dtype=np.int32)
    head_masks = np.zeros((example_num, max_relation_num, max_entity_num), dtype=np.int32)
    tail_masks = np.zeros((example_num, max_relation_num, max_entity_num), dtype=np.int32)
    for idx, example in enumerate(tqdm(examples)):
        titles.append(example["title"])
        sent_num.append(len(example['sents']))
        vertex_set, sentences = convert_single_example(example)
        len_vertex.append(len(vertex_set))
        segment_id = []
        text = []
        cur_seq_length = 0
        len_sentence = []
        for i, sentence in enumerate(sentences):
            text.extend(sentence)
            segment_id.extend([i + 1] * len(sentence))
            if cur_seq_length < max_seq_length:
                len_sentence.append(cur_seq_length)
                sentence_masks[idx, i, cur_seq_length] = 1
            cur_seq_length += len(sentence)
        if cur_seq_length > max_seq_length:
            text = text[:max_seq_length]
            segment_id = segment_id[:max_seq_length]
            seq_mask = [1] * max_seq_length
        else:
            text.extend(["[PAD]"] * (max_seq_length - cur_seq_length))
            segment_id.extend([0] * (max_seq_length - cur_seq_length))
            seq_mask = [1] * cur_seq_length + [0] * (max_seq_length - cur_seq_length)
        tokens.append(text)
        input_id = tokenizer.convert_tokens_to_ids(text)
        input_ids.append(input_id)
        segment_ids.append(segment_id)
        seq_masks.append(seq_mask)

        pair_counter = 0
        for h, e1 in enumerate(vertex_set):
            for mention in e1:
                if mention['sent_id'] < len(len_sentence):
                    mention_pos = len_sentence[mention['sent_id']] + mention['pos'][0]
                    if mention_pos < max_seq_length:
                        entity_masks[idx, h, mention_pos] = 1.0 / len(e1)
            for t, e2 in enumerate(vertex_set):
                if h != t:
                    head_masks[idx, pair_counter, h] = 1
                    tail_masks[idx, pair_counter, t] = 1
                    pair_counter += 1
        relation_masks[idx, : pair_counter] = 1
        attention_array = np.zeros(max_node_num, dtype=np.int32)
        attention_array[: len(sentences)] = 1
        attention_array[max_sentence_num: max_sentence_num + len(vertex_set)] = 1
        attention_masks[idx, : len(sentences)] = attention_array
        attention_masks[idx, max_sentence_num: max_sentence_num + len(vertex_set)] = attention_array

    dataset = {
        "titles": titles,
        "len_vertex": len_vertex,
        "input_ids": np.asarray(input_ids),
        "segment_ids": np.asarray(segment_ids),
        "input_mask": np.asarray(seq_masks),
        "entity_mask": entity_masks,
        "sentence_mask": sentence_masks,
        "attention_mask": attention_masks,
        "relation_mask": relation_masks,
        "head_mask": head_masks,
        "tail_mask": tail_masks,
        "sent_num": sent_num}
    return dataset


def batch_iter(example_num, batch_size, shuffle=False):
    index_list = list(range(example_num))
    if shuffle:
        random.shuffle(index_list)
    batch_num = example_num // batch_size + 1 if example_num % batch_size else example_num // batch_size
    for i in range(batch_num):
        if i + 1 < batch_num:
            cur_batch_index = index_list[i * batch_size: (i + 1) * batch_size]
        else:
            cur_batch_index = index_list[i * batch_size:]
        yield cur_batch_index


def evaluate(logits_list, evi_logits_list, data, index_list_s, index_map):
    print('Evaluation:')
    results = []
    evi_results = []
    evi_truth = []
    labels = data["not_na_labels"]
    len_vertex = data["len_vertex"]
    is_in_train = data["is_in_train"]
    positive_num = 0
    for k, index_list in tqdm(enumerate(index_list_s)):
        logits = logits_list[k]
        for i, index in enumerate(index_list):
            label = labels[index]
            in_train = is_in_train[index]
            positive_num += len(label)
            pair_counter = 0
            for h in range(len_vertex[index]):
                for t in range(len_vertex[index]):
                    if h != t:
                        for r in range(1, len(rel2id)):
                            item_label = (h, t, r) in label
                            item_logit = logits[i, pair_counter, r]
                            item_in_train = (h, t, r) in in_train
                            result = {"logit": item_logit,
                                      "label": item_label,
                                      "in_train": item_in_train,
                                      "rel_id": (index, pair_counter, r)
                                      }
                            results.append(result)
                            if item_label:
                                for sen_index, x in enumerate(data['evidence'][index, pair_counter, r, :]):
                                    if x > 0:
                                        evi_truth.append((index, pair_counter, r, sen_index))
                        pair_counter += 1
    print(f'len of evi truth:{len(evi_truth)}')
    results.sort(key=lambda x: x["logit"], reverse=True)
    precision = []
    recall = []
    true_positive = 0
    for i, item in enumerate(results):
        true_positive += item["label"]
        precision.append(float(true_positive) / (i + 1))
        recall.append(float(true_positive) / positive_num)
    precision = np.asarray(precision, dtype='float32')
    recall = np.asarray(recall, dtype='float32')
    auc = sklearn.metrics.auc(x=recall, y=precision)
    f1 = (2 * precision * recall / (precision + recall + 1e-20))
    max_f1 = f1.max()
    max_f1_pos = f1.argmax()
    threshold = results[max_f1_pos]["logit"]

    ign_p = []
    ign_r = []
    tp = 0
    tp_in_train = 0
    for i, item in enumerate(results):
        tp += item["label"]
        tp_in_train += item["label"] & item["in_train"]
        if tp == tp_in_train:
            ign_p.append(0)
        else:
            ign_p.append(float(tp - tp_in_train) / (i + 1 - tp_in_train))
        ign_r.append(float(tp) / positive_num)
    ign_p = np.asarray(ign_p, dtype='float32')
    ign_r = np.asarray(ign_r, dtype='float32')
    ign_auc = sklearn.metrics.auc(x=ign_r, y=ign_p)
    ign_f1 = (2 * ign_p * ign_r / (ign_p + ign_r + 1e-20))
    max_ign_f1 = ign_f1.max()
    max_ign_f1_pos = ign_f1.argmax()
    ign_threshold = results[max_ign_f1_pos]["logit"]

    print('extract evi result...')
    for each in tqdm(results[:max_ign_f1_pos + 1]):
        rel_id = each['rel_id']
        index = rel_id[0]
        u, v = index_map[index][0], index_map[index][1]
        for sent_idx, x in enumerate(evi_logits_list[u][v, rel_id[1], rel_id[2], :]):
            evi_id = (rel_id[0], rel_id[1], rel_id[2], sent_idx)
            logit = x
            label = evi_id in evi_truth
            evi_result = {
                'evi_id': evi_id,
                'logit': logit,
                'label': label
            }
            evi_results.append(evi_result)
    evi_results.sort(key=lambda x: x["logit"], reverse=True)
    max_evi_f1 = 0
    max_evi_p = 0
    max_evi_r = 0
    evi_threshold = 0
    evi_positive_num = 0
    print('calculate evi f1...')
    for idx, each in tqdm(enumerate(evi_results)):
        evi_positive_num += each['label']
        evi_p = evi_positive_num / (idx + 1)
        evi_r = evi_positive_num / len(evi_truth)
        evi_f1 = (2 * evi_p * evi_r / (evi_p + evi_r + 1e-20))
        if evi_f1 > max_evi_f1:
            max_evi_f1, max_evi_p, max_evi_r = evi_f1, evi_p, evi_r
            evi_threshold = each['logit']

    score = {"f1": max_f1, "auc": auc, "p": precision, "r": recall, "threshold": threshold,
            "ign_f1": max_ign_f1, "ign_auc": ign_auc,
            "ign_p": ign_p, "ign_r": ign_r, "ign_threshold": ign_threshold,
            'evi_p': max_evi_p, 'evi_r': max_evi_r, 'evi_f1': max_evi_f1, 'evi_threshold': evi_threshold}

    return score

def evaluate1(logits_list, data, index_list_s):
    print('evaluation:')
    results = []
    labels = data["not_na_labels"]
    len_vertex = data["len_vertex"]
    is_in_train = data["is_in_train"]
    positive_num = 0
    for k, index_list in enumerate(index_list_s):
        logits = logits_list[k]
        for i, index in enumerate(index_list):
            label = labels[index]
            in_train = is_in_train[index]
            positive_num += len(label)
            pair_counter = 0
            for h in range(len_vertex[index]):
                for t in range(len_vertex[index]):
                    if h != t:
                        for r in range(1, len(rel2id)):
                            item_label = (h, t, r) in label
                            item_logit = logits[i, pair_counter, r]
                            item_in_train = (h, t, r) in in_train
                            result = {"logit": item_logit,
                                      "label": item_label,
                                      "in_train": item_in_train}
                            results.append(result)
                        pair_counter += 1

    results.sort(key=lambda x: x["logit"], reverse=True)
    precision = []
    recall = []
    true_positive = 0
    for i, item in enumerate(results):
        true_positive += item["label"]
        precision.append(float(true_positive) / (i + 1))
        recall.append(float(true_positive) / positive_num)
    precision = np.asarray(precision, dtype='float32')
    recall = np.asarray(recall, dtype='float32')
    auc = sklearn.metrics.auc(x=recall, y=precision)
    f1 = (2 * precision * recall / (precision + recall + 1e-20))
    max_f1 = f1.max()
    max_f1_pos = f1.argmax()
    threshold = results[max_f1_pos]["logit"]

    ign_p = []
    ign_r = []
    tp = 0
    tp_in_train = 0
    for i, item in enumerate(results):
        tp += item["label"]
        tp_in_train += item["label"] & item["in_train"]
        if tp == tp_in_train:
            ign_p.append(0)
        else:
            ign_p.append(float(tp - tp_in_train) / (i + 1 - tp_in_train))
        ign_r.append(float(tp) / positive_num)
    ign_p = np.asarray(ign_p, dtype='float32')
    ign_r = np.asarray(ign_r, dtype='float32')
    ign_auc = sklearn.metrics.auc(x=ign_r, y=ign_p)
    ign_f1 = (2 * ign_p * ign_r / (ign_p + ign_r + 1e-20))
    max_ign_f1 = ign_f1.max()
    max_ign_f1_pos = ign_f1.argmax()
    ign_threshold = results[max_ign_f1_pos]["logit"]
    return {"f1": max_f1, "auc": auc, "p": precision, "r": recall, "threshold": threshold,
            "ign_f1": max_ign_f1, "ign_auc": ign_auc,
            "ign_p": ign_p, "ign_r": ign_r, "ign_threshold": ign_threshold}



def inference(logits, evi_logits, data, index_list, threshold, evi_threshold):
    results = []
    titles = data["titles"]
    len_vertex = data["len_vertex"]
    for i, index in enumerate(index_list):
        pair_counter = 0
        for h in range(len_vertex[index]):
            for t in range(len_vertex[index]):
                if h != t:
                    for r in range(1, len(rel2id)):
                        if logits[index, pair_counter, r] >= threshold:
                            result = {"title": titles[index],
                                      "h_idx": h,
                                      "t_idx": t,
                                      "r": id2rel[r],
                                      "evidence": []}
                            for sent_idx, x in evi_logits[index, pair_counter, r]:
                                if x >= evi_threshold:
                                    result['evidence'].append(sent_idx)
                            results.append(result)
                    pair_counter += 1

    with open("result.json", "w") as jf:
        json.dump(results, jf, ensure_ascii=False)


if __name__ == '__main__':
    train_file = "../DocRED/train_annotated.json"
    dev_file = "../DocRED/dev.json"
    test_file = "../DocRED/test.json"
    me, ms, mr = get_data_property([train_file, dev_file, test_file])
    print("max entity number: {}\nmax sentence number: {}\nmax relation number: {}".format(me, ms, mr))
    train_data = load_data(train_file, me, ms, mr)
    dev_data = load_data(dev_file, me, ms, mr, is_train=False)
    test_file = load_test_data(test_file, me, ms, mr)
