export CUDA_VISIBLE_DEVICES=0,1,2,3

# ViT-B/32
job_name="ex0"
DATA_PATH="../data/VR_Dataset"
SAVE_PATH="log"
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29531\
    main_pau.py --do_train --do_rerank_learn --num_thread_reader=8 \
    --epochs=3 --batch_size=256 --n_display=10 \
    --init_model ${MODEL_PATH} \
    --data_path ${DATA_PATH}/MSVD/msvd_data \
    --features_path ${DATA_PATH}/MSVD/MSVD_Videos \
    --output_dir ${SAVE_PATH} \
    --lr 1e-5 --max_words 32 --max_frames 12 --batch_size_val 300 \
    --datatype msvd \
    --feature_framerate 1 --coef_lr 1e-2 \
    --freeze_layer_num 0 --slice_framepos 2 \
    --tau 5 --K 32 --lambda1 1 --lambda2 20 --lambda3 0.015 \
    --rerank_coe_v 0.05 --rerank_coe_t 0.5 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a log/${job_name}.out