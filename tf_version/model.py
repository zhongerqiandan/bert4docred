import tensorflow as tf
from bert import modeling

class Model(object):
    def __init__(self, max_entity_num, max_sentence_num, max_relation_num, max_seq_length, entity_types,
                 class_num, bert_config, hidden_size, hidden_layers, attention_heads, intermediate_size,
                 hidden_dropout_prob, attention_probs_dropout_prob, graph_hidden_dropout_prob, graph_attention_probs_dropout_prob):
        max_node_num = max_sentence_num + max_entity_num

        self.input_ids = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int32, name="input_ids")
        self.input_mask = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int32, name="input_mask")
        self.segment_ids = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int32, name="segment_ids")
        self.entity_types = tf.placeholder(shape=[None, max_node_num], dtype=tf.int32, name="entity_types")
        self.entity_mask = tf.placeholder(
            shape=[None, max_entity_num, max_seq_length], dtype=tf.float32, name="entity_mask")
        self.sentence_mask = tf.placeholder(
            shape=[None, max_sentence_num, max_seq_length], dtype=tf.float32, name="sentence_mask")
        self.relation_mask = tf.placeholder(
            shape=[None, max_relation_num], dtype=tf.float32, name="relation_mask")
        self.attention_mask = tf.placeholder(
            shape=[None, max_node_num, max_node_num], dtype=tf.float32, name="graph_mask")
        self.head_mask = tf.placeholder(
            shape=[None, max_relation_num, max_entity_num], dtype=tf.float32, name="head_mask")
        self.tail_mask = tf.placeholder(
            shape=[None, max_relation_num, max_entity_num], dtype=tf.float32, name="tail_mask")
        self.multi_labels = tf.placeholder(
            shape=[None, max_relation_num, class_num], dtype=tf.int32, name="multi_labels")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")


        self.hidden_dropout_prob = tf.cond(
            self.is_training, lambda: hidden_dropout_prob, lambda: 0.0)
        self.attention_probs_dropout_prob = tf.cond(
            self.is_training, lambda: attention_probs_dropout_prob, lambda: 0.0)

        self.graph_hidden_dropout_prob = tf.cond(
            self.is_training, lambda: graph_hidden_dropout_prob, lambda: 0.0)
        self.graph_attention_probs_dropout_prob = tf.cond(
            self.is_training, lambda: graph_attention_probs_dropout_prob, lambda: 0.0)

        self.entity_type_embedding = tf.get_variable(
            shape=[entity_types, 32], dtype=tf.float32, name="entity_type_embedding")
        self.entity_type_rep = tf.nn.embedding_lookup(
            self.entity_type_embedding, self.entity_types)

        self.seq_rep = self.bert_encoder(
            bert_config, self.hidden_dropout_prob, self.attention_probs_dropout_prob,
            self.input_ids, self.input_mask, self.segment_ids)
        self.entity_rep = tf.matmul(self.entity_mask, self.seq_rep)
        self.sentence_rep = tf.matmul(self.sentence_mask, self.seq_rep)
        self.graph_rep = tf.concat([self.entity_rep, self.sentence_rep], axis=1)
        self.graph_rep = tf.concat([self.graph_rep, self.entity_type_rep], axis=-1)
        self.graph_rep = tf.layers.dense(self.graph_rep, hidden_size, tf.nn.relu)
        self.final_rep = modeling.transformer_model(
            input_tensor=self.graph_rep,
            attention_mask=self.attention_mask,
            hidden_size=hidden_size,
            num_hidden_layers=hidden_layers,
            num_attention_heads=attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=self.graph_hidden_dropout_prob,
            attention_probs_dropout_prob=self.graph_attention_probs_dropout_prob)
        self.entity_rep = self.final_rep[:, : max_entity_num]
        self.head_rep = tf.matmul(self.head_mask, self.entity_rep)
        self.tail_rep = tf.matmul(self.tail_mask, self.entity_rep)
        bi_hidden_size = self.head_rep.get_shape().as_list()[-1]
        self.logits = self.bilinear_function(self.head_rep, self.tail_rep, bi_hidden_size, class_num)
        self.sigmoid = tf.sigmoid(self.logits, name='sigmoid')
        self.entropy = tf.losses.sigmoid_cross_entropy(
            self.multi_labels, self.logits, reduction=tf.losses.Reduction.NONE)
        self.loss = tf.reduce_sum(
            tf.multiply(self.entropy, tf.expand_dims(self.relation_mask, axis=-1))
        ) / tf.reduce_sum(self.relation_mask)

    def bilinear_function(self, x1, x2, in_size, out_size):
        with tf.variable_scope('bilinear'):
            weight = tf.get_variable(
                shape=[out_size, in_size, in_size], dtype=tf.float32, name='weight')
            bias = tf.get_variable(shape=[out_size], dtype=tf.float32, name='bias')
        outputs = []
        for i in range(out_size):
            buff = tf.matmul(x1, weight[i])
            buff = tf.multiply(buff, x2)
            buff = tf.reduce_sum(buff, axis=-1, keep_dims=True)
            outputs.append(buff)
        outputs = tf.concat(outputs, axis=-1)
        outputs += bias
        return outputs

    def diag_bilinear(self, x1, x2, in_size, out_size):
        weight = tf.get_variable(
            shape=[out_size, in_size], dtype=tf.float32, name="diag_bilinear_weight")
        bias = tf.get_variable(shape=[out_size], dtype=tf.float32, name="diag_bilinear_bias")
        outputs = []
        for i in range(out_size):
            buff = tf.multiply(x1, weight[i])
            buff = tf.multiply(buff, x2)
            buff = tf.reduce_sum(buff, axis=-1, keep_dims=True)
            outputs.append(buff)
        outputs = tf.concat(outputs, axis=-1)
        outputs += bias
        return outputs

    def bert_encoder(self, bert_config, hidden_dropout_prob, attention_probs_dropout_prob,
                     input_ids, input_mask, segment_ids):
        encoder = modeling.BertModel(
            config=bert_config,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
        return encoder.sequence_output