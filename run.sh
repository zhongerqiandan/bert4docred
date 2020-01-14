set -e
export CUDA_VISIBLE_DEVICES=0
export PROJECT_PATH=/home/jiangweiwei/project/bert4docred
export DATA_PATH=$PROJECT_PATH/data
export CHECKPOINT_DIR=/home/jiangweiwei/project/bert4docred/checkpoint
export BERT_CONFIG_PATH=/data/jiangweiwei/bertmodel/cased_L-12_H-768_A-12/bert_config.json
export BERT_CHECKPOINT_PATH=/data/jiangweiwei/bertmodel/cased_L-12_H-768_A-12/bert_model.ckpt
export BERT_VOCAB_PATH=/data/jiangweiwei/bertmodel/cased_L-12_H-768_A-12/vocab.txt
export EXPORT_DIR=$CHECKPOINT_DIR/export
#mkdir -p $CHECKPOINT_DIR
/home/huyong/miniconda3/bin/python run_docred.py \
  --data_path=$DATA_PATH \
  --vocab_path=$BERT_VOCAB_PATH \
  --bert_config_path=$BERT_CONFIG_PATH \
  --bert_checkpoint_path=$BERT_CHECKPOINT_PATH \
  --epochs=10 \
  --train_or_test=test \
  --checkpoint_path=$CHECKPOINT_DIR/best.weights \
  --mode=all \
  --predict_path=$PROJECT_PATH/result