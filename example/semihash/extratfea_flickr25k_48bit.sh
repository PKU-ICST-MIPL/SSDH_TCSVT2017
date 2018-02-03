#!/usr/bin/env sh

RUN_BIN=./build/tools

PROTO=./examples/semihash/configure/flickr25k/flickr25k_semihash_feature_vggf_48b.prototxt
TOTAL_NUM=20000
BATCH_NUM=200
#FEA_NUM=`expr $TOTAL_NUM / $BATCH_NUM + 1`
FEA_NUM=`expr $TOTAL_NUM / $BATCH_NUM`
GPU_ID=0
echo "Begin Extract fea"

MODEL_NAME=models/flickr25k/48bit/vggf_semirank_th_ak_pl_sg_ftnin/flickr25k_semihash_iter_10000.caffemodel
FEA_DIR=features/flickr25k/48bit_features_vggf_semirank_th_ak_pl_sg_ftnin_10K/
echo $MODEL_NAME
echo $FEA_DIR
echo $PROTO
echo "Total Feature num: ${FEA_NUM}"
GLOG_logtostderr=0
${RUN_BIN}/extract_features.bin ${MODEL_NAME} ${PROTO} prob ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID

