# default
MODEL_NAME=convnext_base_in22k
MODEL_PATH=/home/lihp/prj/DRCT/output/DRCT-2M/2/convnext_base_in22k_224_drct_amp_crop/weights/last_acc0.9998.pth
DEVICE_ID=5
EMBEDDING_SIZE=1024
MODEL_NAME=${1:-$MODEL_NAME}
MODEL_PATH=${2:-$MODEL_PATH}
DEVICE_ID=${3:-$DEVICE_ID}
EMBEDDING_SIZE=${4:-$EMBEDDING_SIZE}
ROOT_PATH=/data3/lihp/DRCT/dataset/AIGC_data/DRCT_data/MSCOCO
FAKE_ROOT_PATH=/data3/lihp/DRCT/dataset/AIGC_data/DRCT_data/DRCT-2M/images
DATASET_NAME=DRCT-2M
SAVE_TXT=../output/results/DRCT-2M_metrics.txt
INPUT_SIZE=224
BATCH_SIZE=24
FAKE_INDEXES=(1)
#FAKE_INDEXES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
for FAKE_INDEX in ${FAKE_INDEXES[@]}
do
  echo FAKE_INDEX:${FAKE_INDEX}
  python fgsm_train.py --root_path ${ROOT_PATH} --fake_root_path ${FAKE_ROOT_PATH} --model_name ${MODEL_NAME} \
                  --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
                  --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} \
                  --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE}
done
