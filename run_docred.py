# -*- coding: utf-8 -*-
# @Time    : 2020/1/12 7:55 下午
# @Author  : Jiangweiwei
# @mail    : zhongerqiandan@163.com

import argparse
import tensorflow as tf
import os
import time
import keras.backend.tensorflow_backend as KTF
import pickle
import json
from gen_data import build_data
import numpy as np
from tqdm import tqdm
from random import shuffle
from keras import Input, Model
from keras import backend as K
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Dense, Flatten
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint
from tokenization import FullTokenizer
from sklearn.metrics import classification_report
from sklearn import metrics


class FscoreEvaluate(Callback):
    def __init__(self, x, y, save_path):
        self.F1 = []
        self.best = 0.
        self.x = x
        self.y = y
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        f1 = self.evaluate()
        print(f'epoch:{epoch},f1_score in dev set:{f1}')
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            self.model.save_weights(self.save_path)

    def evaluate(self):
        predict = []
        target = np.argmax(self.y, -1)
        input_ids_array, segment_ids_array, h_t_pos_array = self.x[0], self.x[1], self.x[2]
        for each in zip(input_ids_array, segment_ids_array, h_t_pos_array):
            input_ids, segment_ids, h_t_pos = each[0].reshape(1, -1), each[1].reshape(1, -1), each[2].reshape(1, 2, -1)
            multi_prob = self.predict([input_ids, segment_ids, h_t_pos])
            pre_rel_id = np.argmax(multi_prob)
            predict.append(pre_rel_id)
        f1 = metrics.f1_score(target, predict, average='weighted')
        return f1

    def predict(self, x):
        '''

        :param x: x = [index, segment_ids, h_t_pos]
        :return:
        '''
        multi_prob = self.model.predict(x)
        return list(multi_prob[0])


class LossEvaluate(Callback):
    def __init__(self, save_path):
        self.lowest = 1e10
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            self.model.save_weights(self.save_path)


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


class BERTDocred:
    def __init__(self, args):
        self.setting = args
        self.model = self._build_model()

    def _build_model(self):
        bert_model = load_trained_model_from_checkpoint(self.setting['bert_config_path'],
                                                        self.setting['bert_checkpoint_path'],
                                                        seq_len=self.setting['max_seq_len'])
        for l in bert_model.layers:
            l.trainable = True
        print('bert model')
        print(bert_model.summary())

        x1 = Input(shape=(self.setting['max_seq_len'],), batch_shape=(None, self.setting['max_seq_len']),
                   name='input_ids')
        x2 = Input(shape=(self.setting['max_seq_len'],), batch_shape=(None, self.setting['max_seq_len']),
                   name='segment_ids')
        x3 = Input(batch_shape=(None, 2, self.setting['max_seq_len']),
                   name='h_and_t_pos')
        x = bert_model([x1, x2])
        h_t_tensor = Lambda(lambda x: tf.matmul(x[0], x[1]))([x3, x])
        h_t_tensor_concat = Flatten()(h_t_tensor)
        multi_prob = Dense(self.setting['num_rel'], activation='sigmoid')(h_t_tensor_concat)
        model = Model([x1, x2, x3], [multi_prob])
        print(model.summary())
        return model

    def train_model(self, train_x, train_y, dev_x=None, dev_y=None):
        self.model.compile(loss='binary_crossentropy',
                           optimizer=Adam(self.setting['lr']),
                           metrics=['accuracy'])
        # if self.setting['continue_checkpoint_path']:
        #     print(f"load weight from {self.setting['continue_checkpoint_path']}")
        #     self.model.load_weights(self.setting['continue_checkpoint_path'])
        if self.setting['mode'] == 'all':
            evaluator = LossEvaluate(self.setting['checkpoint_path'])
            self.model.fit(train_x,
                           train_y,
                           batch_size=self.setting['train_batch_size'],
                           epochs=self.setting['epochs'],
                           shuffle=False,
                           verbose=1,
                           callbacks=[evaluator])
        else:
            evaluator = FscoreEvaluate(dev_x, dev_y, self.setting['checkpoint_path'])
            self.model.fit(train_x,
                           train_y,
                           batch_size=self.setting['train_batch_size'],
                           epochs=self.setting['epochs'],
                           shuffle=False,
                           verbose=1,
                           callbacks=[evaluator])

    def load_model_and_predict(self, test_data):
        model = self.model
        model.load_weights(self.setting['checkpoint_path'])
        test_data.sort(key=lambda x: x['id'])
        sample_ids = []
        for d in test_data:
            sample_ids.append(d['id'])
        sample_ids = list(set(sample_ids))
        sample_ids.sort()
        result = []
        error_ids = []
        print(f'test data its:{len(test_data)}')
        p = 0
        for _id in tqdm(sample_ids):
            temp_sample = []
            for i in range(p, len(test_data)):
                if test_data[i]['id'] != _id:
                    p = i
                    break
                else:
                    temp_sample.append(test_data[i])
            # print(f'index:{p-len(temp_sample)}:{p-1}')
            input_ids_array, segment_ids_array, h_t_pos_array = [], [], []
            for each in temp_sample:
                input_ids_array.append(each['input_ids'])
                segment_ids_array.append(each['segment_ids'])
                h_t_pos_array.append(each['h_t_pos'])
            input_ids_array, segment_ids_array, h_t_pos_array = np.array(input_ids_array), np.array(
                segment_ids_array), np.array(h_t_pos_array)
            try:
                predict_probs = model.predict([input_ids_array, segment_ids_array, h_t_pos_array])
            except:
                error_ids.append(_id)
                continue
            predict_probs = predict_probs.tolist()
            predict_label = list(map(lambda x: (x.index(max(x)), max(x)), predict_probs))
            candidate_element = max(predict_label, key=lambda x: x[1])
            # if candidate_element[1] < self.setting['threshold']:
            #     continue
            # else:
            #     candidate_index = predict_label.index(candidate_element)
            #     sample = temp_sample[candidate_index]
            #     title = sample['title']
            #     h_idx = sample['h_idx']
            #     t_idx = sample['t_idx']
            #     rel_id = candidate_element[0]
            #     evidence = sample['evidence']
            #     d = {'title': title, 'h_idx': h_idx, 't_idx': t_idx, 'r': rel_id, 'evidence': evidence}
            candidate_index = predict_label.index(candidate_element)
            sample = temp_sample[candidate_index]
            title = sample['title']
            h_idx = sample['h_idx']
            t_idx = sample['t_idx']
            rel_id = candidate_element[0]
            evidence = sample['evidence']
            d = {'title': title, 'h_idx': h_idx, 't_idx': t_idx, 'r': rel_id, 'evidence': evidence,
                 'prob': candidate_element[1]}

            '''
            predict_label:[(rel_id, prob), (rel_id,prob),...], 对应同一个id'i-j-k'的样本
            '''
            result.append(d)
        with open(os.path.join(self.setting['predict_path'], 'result_for_search_threshold_100_200.json'), 'w') as f:
            json.dump(result, f)
        with open(os.path.join(self.setting['predict_path'], 'error_ids_100_200.txt'), 'w') as f:
            f.write('\n'.join(error_ids))
        print(f'save predic file in {self.setting["predict_path"]}')

    def export_pb_format(self):
        model = self.model
        model.load_weights(self.setting['checkpoint_path'])
        signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={'input_ids': model.input[0],
                    'segment_ids': model.input[1],
                    'h_and_t_pos': model.input[2],
                    },
            outputs={'logits': model.output})
        export_path = os.path.join(
            tf.compat.as_bytes(self.setting['export_dir']),
            tf.compat.as_bytes(str(int(time.time()))))
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess=K.get_session(),
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict':
                    signature,
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature,

            },
            legacy_init_op=legacy_init_op)
        builder.save()


def build_data_set(data_file_name, rel2id, tokenizer, mode, num_classes):
    '''

    :param data_file_name:
    :param rel2id:
    :param tokenizer:
    :param mode: train, dev, or all, if mode == 'all', data_file_name = [train_file, dev_file]
    :param num_classes:
    :return:
    '''

    def one_hot(position, num_classes):
        one_hot_label = [0.] * num_classes
        one_hot_label[position] = 1.
        return one_hot_label

    data_set = build_data(data_file_name, rel2id, tokenizer, mode)
    # data_set = data_set[:100]
    if mode == 'train' or mode == 'dev' or mode == 'all':
        if mode == 'all':
            data_set[0] = build_data(data_file_name[0], rel2id, tokenizer, mode)
            data_set[1] = build_data(data_file_name[1], rel2id, tokenizer, mode)
            data_set = data_set[0] + data_set[1]
            shuffle(data_set)
        else:
            data_set = build_data(data_file_name, rel2id, tokenizer, mode)
        input_ids_array = []
        segment_ids_array = []
        h_t_pos_array = []
        one_hot_rel_id_array = []
        for data in tqdm(data_set):
            input_ids_array.append(data['input_ids'])
            segment_ids_array.append(data['segment_ids'])
            h_t_pos_array.append(data['h_t_pos'])
            one_hot_rel_id_array.append(one_hot(data['rel_id'], num_classes))
        return [np.array(input_ids_array), np.array(segment_ids_array), np.array(h_t_pos_array)], np.array(
            one_hot_rel_id_array)
    else:
        '''
        each:'id':str,'title':str, 'h_idx','t_idx', 'input_ids':[], 'segment_ids':[], 'h_t_pos':[[], []], 'evidence':[]
        '''
        return data_set


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    session = tf.Session(config=config)
    KTF.set_session(session)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="数据集路径")
    parser.add_argument("--vocab_path", default=None, type=str, required=True,
                        help="词典路径")
    parser.add_argument("--bert_config_path", default=None, type=str, required=True,
                        help="BERT设置路径")
    parser.add_argument("--bert_checkpoint_path", default=None, type=str, required=True,
                        help="BERT checkpoint路径")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="最大长度")
    # 训练参数
    parser.add_argument("--train_or_test", default='train', type=str, choices=['train', 'test'],
                        help='训练或者预测')
    parser.add_argument("--train_batch_size", default=6, type=int,
                        help="训练阶段batch size")
    parser.add_argument("--lr", default=3e-5, type=float,
                        help="学习率")
    parser.add_argument("--epochs", default=3, type=int,
                        help="训练的epoch数")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True,
                        help="模型保存的路径")
    parser.add_argument("--predict_path", default=None, type=str,
                        help="预测结果保存路径")
    # parser.add_argument("--export_dir", default=None, type=str,
    #                     help="tensorflow serving模型")
    parser.add_argument("--num_rel", default=97, type=int,
                        help="类别总数")
    parser.add_argument("--threshold", default=0.95, type=int,
                        help="阈值")
    parser.add_argument("--mode", default='train', type=str, choices=['train', 'dev', 'all'],
                        help='')

    args = parser.parse_args().__dict__
    tokenizer = FullTokenizer(args['vocab_path'], do_lower_case=False)
    doc_tokenizer = DocRedTokenizer(tokenizer, args['max_seq_len'])
    rel2id = json.load(open(os.path.join(args['data_path'], 'rel2id.json'), "r"))
    docred_model = BERTDocred(args)
    if args['train_or_test'] == 'train':
        if args['mode'] == 'all':
            train_x, train_y = build_data_set(
                [os.path.join(args['data_path'], 'train_annotated.json'), os.path.join(args['data_path'], 'dev.json')],
                rel2id,
                doc_tokenizer, args['mode'], args['num_rel'])
            docred_model.train_model(train_x, train_y)
        else:
            train_x, train_y = build_data_set(os.path.join(args['data_path'], 'train_annotated.json'), rel2id,
                                              doc_tokenizer, 'train', args['num_rel'])
            dev_x, dev_y = build_data_set(os.path.join(args['data_path'], 'dev.json'), rel2id,
                                          doc_tokenizer, 'dev', args['num_rel'])
            docred_model.train_model(train_x, train_y, dev_x, dev_y)
    else:
        test_data = build_data_set(os.path.join(args['data_path'], 'dev.json'), rel2id, doc_tokenizer, 'test',
                                   args['num_rel'])
        docred_model.load_model_and_predict(test_data)
