import os
import json
import numpy as np
import tensorflow as tf
import utils
from tqdm import tqdm
from model import Model
from bert import modeling, optimization

tf.flags.DEFINE_integer('batch_size', 4, "batch size")
tf.flags.DEFINE_integer('hidden_size', 320, "hidden size of graph")
tf.flags.DEFINE_integer('hidden_layers', 6, "number of graph layers")
tf.flags.DEFINE_integer('attention_heads', 32, "number of attention heads")
tf.flags.DEFINE_integer('intermediate_size', 2048, "intermediate size of graph")
tf.flags.DEFINE_integer('max_seq_length', 512, "max length of sequence")
tf.flags.DEFINE_integer('max_epoch', 60, "max training epoch")
tf.flags.DEFINE_float('learning_rate', 5e-5, "learning rate")
tf.flags.DEFINE_float('graph_hidden_dropout_prob', 0.1, "graph_hidden_dropout_prob")
tf.flags.DEFINE_float('graph_attention_probs_dropout_prob', 0.1, "graph_attention_probs_dropout_prob")
tf.flags.DEFINE_float('test_threshold', 0.5, "test threshold")
tf.flags.DEFINE_boolean('do_train', True, "do training or inference")

FLAGS = tf.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
checkpoint_dir = "checkpoint"
bert_dir = "/data/jiangweiwei/bertmodel/uncased_L-12_H-768_A-12"
bert_checkpoint = os.path.join(bert_dir, "bert_model.ckpt")
bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, "bert_config.json"))
# init_checkpoint = '/home/jiangweiwei/project/bert4docred/best_rel/ign_model'

with open("../data/ner2id.json", 'r') as jf:
    ner2id = json.load(jf)
with open("../data/rel2id.json", 'r') as jf:
    rel2id = json.load(jf)

train_file = "../data/train_annotated.json"
dev_file = "../data/dev.json"
test_file = "../data/test.json"
me, ms, mr = utils.get_data_property([train_file, dev_file, test_file])
print("max entity number: {}\nmax sentence number: {}\nmax relation number: {}".format(me, ms, mr))



def train():
    print("Loading data...")
    train_data = utils.load_data(train_file, me, ms, mr)
    dev_data = utils.load_data(dev_file, me, ms, mr, is_train=False)

    train_example_num = len(train_data["input_ids"])
    dev_example_num = len(dev_data["input_ids"])
    print("Done.")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.Session(config=config) as sess:
        model = Model(max_entity_num=me,
                      max_sentence_num=ms,
                      max_relation_num=mr,
                      max_seq_length=FLAGS.max_seq_length,
                      entity_types=len(ner2id),
                      class_num=len(rel2id),
                      bert_config=bert_config,
                      hidden_size=FLAGS.hidden_size,
                      hidden_layers=FLAGS.hidden_layers,
                      attention_heads=FLAGS.attention_heads,
                      intermediate_size=FLAGS.intermediate_size,
                      hidden_dropout_prob=bert_config.hidden_dropout_prob,
                      attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
                      graph_hidden_dropout_prob=FLAGS.graph_hidden_dropout_prob,
                      graph_attention_probs_dropout_prob=FLAGS.graph_attention_probs_dropout_prob,
                      )

        num_train_steps = int(train_data["input_ids"].shape[0] / FLAGS.batch_size * FLAGS.max_epoch)
        train_op = optimization.create_optimizer(loss=model.loss,
                                                 init_lr=FLAGS.learning_rate,
                                                 num_train_steps=num_train_steps,
                                                 num_warmup_steps=int(0.1 * num_train_steps),
                                                 use_tpu=False,
                                                 freeze=False)


        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(max_to_keep=3)
        # bert_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bert')
        # bert_saver = tf.train.Saver(bert_vars)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)
            print('*' * 36)
        else:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, bert_checkpoint)
            tf.train.init_from_checkpoint(bert_checkpoint, assignment_map)
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.global_variables_initializer())
            # tvars = tf.trainable_variables()
            # init_vars = []
            # for v in tvars:
            #     if 'bert' in v.name:
            #         init_vars.append(v)
            # init_saver = tf.train.Saver(var_list=init_vars)
            # init_saver.restore(sess, init_checkpoint)
            print('-' * 36)
        best_ign_f1 = 0
        best_ign_epoch = 0
        for epoch in range(FLAGS.max_epoch):

            print("\nepoch: {}".format(epoch + 1))
            train_losses = []
            for batch_index in tqdm(utils.batch_iter(train_example_num, FLAGS.batch_size, True)):
                feed_dict = {model.input_ids: train_data["input_ids"][batch_index],
                             model.input_mask: train_data["input_mask"][batch_index],
                             model.segment_ids: train_data["segment_ids"][batch_index],
                             model.entity_types: train_data["entity_types"][batch_index],
                             model.entity_mask: train_data["entity_mask"][batch_index],
                             model.sentence_mask: train_data["sentence_mask"][batch_index],
                             model.attention_mask: train_data["attention_mask"][batch_index],
                             model.relation_mask: train_data["relation_mask"][batch_index],
                             model.head_mask: train_data["head_mask"][batch_index],
                             model.tail_mask: train_data["tail_mask"][batch_index],
                             model.multi_labels: train_data["multi_labels"][batch_index],
                             model.is_training: True}
                _, loss = sess.run([train_op, model.loss], feed_dict)
                train_losses.append(loss)
            train_loss = np.mean(train_losses)
            print("train | loss: {:3.4f}".format(train_loss))
            if epoch == 29:
                saver.save(sess, os.path.join(checkpoint_dir, "ign_model"))
            if epoch > 29 :
            # if 1:
                dev_losses = []
                dev_logits = []
                dev_index = []
                print("evaluation:")
                for batch_index in tqdm(utils.batch_iter(dev_example_num, FLAGS.batch_size, False)):
                    feed_dict = {model.input_ids: dev_data["input_ids"][batch_index],
                                 model.input_mask: dev_data["input_mask"][batch_index],
                                 model.segment_ids: dev_data["segment_ids"][batch_index],
                                 model.entity_types: dev_data["entity_types"][batch_index],
                                 model.entity_mask: dev_data["entity_mask"][batch_index],
                                 model.sentence_mask: dev_data["sentence_mask"][batch_index],
                                 model.attention_mask: dev_data["attention_mask"][batch_index],
                                 model.relation_mask: dev_data["relation_mask"][batch_index],
                                 model.head_mask: dev_data["head_mask"][batch_index],
                                 model.tail_mask: dev_data["tail_mask"][batch_index],
                                 model.multi_labels: dev_data["multi_labels"][batch_index],
                                 model.is_training: False}
                    logit, loss = sess.run([model.sigmoid, model.loss], feed_dict)
                    dev_losses.append(loss)
                    dev_logits.append(logit)
                    dev_index.append(batch_index)
                dev_loss = np.mean(dev_losses)
                dm = utils.evaluate1(dev_logits, dev_data, dev_index)
                print("dev | loss: {:3.4f}".format(dev_loss))
                print("F1: {:3.4f} | AUC: {:3.4f} | threshold: {:3.4f}".format(
                    dm["f1"], dm["auc"], dm["threshold"]))
                print("ign_F1: {:3.4f} | ign_AUC: {:3.4f} | ign_threshold: {:3.4f}".format(
                    dm["ign_f1"], dm["ign_auc"], dm["ign_threshold"]))

                with open('exp.txt', 'a') as f:
                    f.write(f'new :')
                    f.write('\n')
                    f.write(f'epoch:{epoch+1}')
                    f.write('\n')
                    f.write("ign_F1: {:3.4f} | ign_AUC: {:3.4f} | ign_threshold: {:3.4f}".format(
                    dm["ign_f1"], dm["ign_auc"], dm["ign_threshold"]))
                    f.write('\n')
                # if dm["f1"] > best_f1:
                #     saver.save(sess, os.path.join(checkpoint_dir, "model"))
                #     best_f1 = dm["f1"]
                #     best_epoch = epoch + 1

                if dm["ign_f1"] > best_ign_f1:
                    saver.save(sess, os.path.join(checkpoint_dir, "ign_model"))
                    best_ign_f1 = dm["ign_f1"]
                    best_ign_epoch = epoch + 1
            # print("epoch: {} | Best F1: {:3.4f}".format(best_epoch, best_f1))
        print("epoch: {} | Best ignore F1: {:3.4f}".format(best_ign_epoch, best_ign_f1))

def infer():
    print("Loading data...")
    dev_data = utils.load_data(test_file, me, ms, mr, is_train=False)

    dev_example_num = len(dev_data["input_ids"])
    print("Done.")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.Session(config=config) as sess:
        model = Model(max_entity_num=me,
                      max_sentence_num=ms,
                      max_relation_num=mr,
                      max_seq_length=FLAGS.max_seq_length,
                      entity_types=len(ner2id),
                      class_num=len(rel2id),
                      bert_config=bert_config,
                      hidden_size=FLAGS.hidden_size,
                      hidden_layers=FLAGS.hidden_layers,
                      attention_heads=FLAGS.attention_heads,
                      intermediate_size=FLAGS.intermediate_size,
                      hidden_dropout_prob=bert_config.hidden_dropout_prob,
                      attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
                      graph_hidden_dropout_prob=FLAGS.graph_hidden_dropout_prob,
                      graph_attention_probs_dropout_prob=FLAGS.graph_attention_probs_dropout_prob,
                      )

        saver = tf.train.Saver(max_to_keep=3)

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)
            print('*' * 36)


        dev_logits = []
        dev_index = []
        print("evaluation:")
        for batch_index in tqdm(utils.batch_iter(dev_example_num, FLAGS.batch_size, False)):
            feed_dict = {model.input_ids: dev_data["input_ids"][batch_index],
                         model.input_mask: dev_data["input_mask"][batch_index],
                         model.segment_ids: dev_data["segment_ids"][batch_index],
                         model.entity_types: dev_data["entity_types"][batch_index],
                         model.entity_mask: dev_data["entity_mask"][batch_index],
                         model.sentence_mask: dev_data["sentence_mask"][batch_index],
                         model.attention_mask: dev_data["attention_mask"][batch_index],
                         model.relation_mask: dev_data["relation_mask"][batch_index],
                         model.head_mask: dev_data["head_mask"][batch_index],
                         model.tail_mask: dev_data["tail_mask"][batch_index],
                         model.is_training: False}
            logit = sess.run([model.sigmoid], feed_dict)
            # print('logit[0] type:', type(logit[0]))
            logit = np.array(logit).tolist()
            dev_logits.append(logit)
            dev_index.append(batch_index)
        index_map = {}
        for u, batch_index in enumerate(dev_index):
            for v, index in enumerate(batch_index):
                index_map[index] = [u, v]

        d = {
            'logts_list': dev_logits,
            'index_list': dev_index,
            'index_map': index_map
        }
        with open('test_logits.json', 'w') as f:
            json.dump(d, f)



def test(threshold, model_name='model'):
    print("Loading data...")
    test_data = utils.load_test_data(test_file, me, ms, mr)
    test_example_num = len(test_data["input_ids"])
    print("Done.")

    with tf.Session() as sess:
        model = Model(max_entity_num=me,
                      max_sentence_num=ms,
                      max_relation_num=mr,
                      max_seq_length=FLAGS.max_seq_length,
                      class_num=len(rel2id),
                      entity_types=len(ner2id),
                      bert_config=bert_config,
                      hidden_size=FLAGS.hidden_size,
                      hidden_layers=FLAGS.hidden_layers,
                      attention_heads=FLAGS.attention_heads,
                      intermediate_size=FLAGS.intermediate_size,
                      hidden_dropout_prob=bert_config.hidden_dropout_prob,
                      attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
                      graph_hidden_dropout_prob=FLAGS.graph_hidden_dropout_prob,
                      graph_attention_probs_dropout_prob=FLAGS.graph_attention_probs_dropout_prob,
                      )

        saver = tf.train.Saver()
        checkpoint = os.path.join(checkpoint_dir, model_name)
        saver.restore(sess, checkpoint)

        test_logits = []
        test_index = []
        for batch_index in tqdm(utils.batch_iter(test_example_num, FLAGS.batch_size, False)):
            feed_dict = {model.input_ids: test_data["input_ids"][batch_index],
                         model.input_mask: test_data["input_mask"][batch_index],
                         model.segment_ids: test_data["segment_ids"][batch_index],
                         model.entity_mask: test_data["entity_mask"][batch_index],
                         model.entity_types: test_data["entity_types"][batch_index],
                         model.sentence_mask: test_data["sentence_mask"][batch_index],
                         model.attention_mask: test_data["attention_mask"][batch_index],
                         model.relation_mask: test_data["relation_mask"][batch_index],
                         model.head_mask: test_data["head_mask"][batch_index],
                         model.tail_mask: test_data["tail_mask"][batch_index],
                         model.is_training: False}
            logit = sess.run(model.sigmoid, feed_dict)
            test_logits.append(logit)
            test_index += batch_index
        test_logits = np.concatenate(test_logits, axis=0)

    utils.inference(test_logits, test_data, test_index, threshold)


if __name__ == '__main__':
    if FLAGS.do_train:
        train()
    else:
        test(threshold=FLAGS.test_threshold)
    # infer()
