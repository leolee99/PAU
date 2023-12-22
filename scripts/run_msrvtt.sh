export CUDA_VISIBLE_DEVICES=0,1,2,3

job_name="ex0"
DATA_PATH="../data/VR_Dataset"
SAVE_PATH="log"
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29571\
    main_pau.py --do_train --num_thread_reader=10 \
    --lr 1e-4 --batch_size=128  --batch_size_val 40 \
    --epochs=5  --n_display=10 \
    --train_csv ${DATA_PATH}/MSRVTT/msrvtt_data/MSRVTT_train.9k.csv \
    --val_csv ${DATA_PATH}/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv \
    --data_path ${DATA_PATH}/MSRVTT/msrvtt_data/MSRVTT_data.json \
    --features_path ${DATA_PATH}/MSRVTT/videos/all \
    --output_dir ${SAVE_PATH} \
    --max_words 32 --max_frames 12 \
    --datatype msrvtt --expand_msrvtt_sentences  \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --tau 5 --K 8 --lambda1 1 --lambda2 100 --lambda3 0.025 \
    --rerank_coe_v 0.05 --rerank_coe_t 0.05 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a log/${job_name}.out
