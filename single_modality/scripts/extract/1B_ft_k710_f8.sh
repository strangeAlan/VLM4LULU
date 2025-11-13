export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='1B_ft_k710_f8'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
DATA_PATH='k710'
MODEL_PATH='pth/1B_ft_k710_f8.pth'

GPUS=1

python extrac_feature.py \
--model internvideo2_1B_patch14_224 \
--data_path ${DATA_PATH} \
--data_set 'Kinetics_sparse' \
--split ',' \
--nb_classes 710 \
--finetune ${MODEL_PATH} \
--log_dir ${OUTPUT_DIR} \
--output_dir ${OUTPUT_DIR} \
--steps_per_print 10 \
--batch_size 8 \
--num_sample 2 \
--input_size 224 \
--short_side_size 224 \
--save_ckpt_freq 100 \
--num_frames 8 \
--sampling_rate 8 \
--num_workers 12 \
--warmup_epochs 2 \
--tubelet_size 1 \
--epochs 9 \
--lr 5e-5 \
--drop_path 0.3 \
--layer_decay 0.9 \
--use_checkpoint \
--checkpoint_num 6 \
--layer_scale_init_value 1e-5 \
--opt adamw \
--opt_betas 0.9 0.999 \
--weight_decay 0.05 \
--test_num_segment 1 \
--test_num_crop 1 \
--zero_stage 1 \
--log_dir ${OUTPUT_DIR} \
--output_dir ${OUTPUT_DIR} \
--test_best \
--extract_features \
--feature_save_dir ${OUTPUT_DIR}/features \
--feature_dtype bfloat16 \
--bf16 \
--feature_split test 