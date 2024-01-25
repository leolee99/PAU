export CUDA_VISIBLE_DEVICES=4

job_name="ex0"
DATA_PATH="../data/VR_Dataset"
SAVE_PATH="log"
MODEL_PATH="log/MSVD-pytorch_model.bin.0"
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29508\
    main_pau.py --do_eval --do_rerank_learn --num_thread_reader=10 \
    --epochs=5 --batch_size=256 --n_display=50 \
    --init_model ${MODEL_PATH} \
    --data_path ${DATA_PATH}/MSVD/msvd_data \
    --features_path ${DATA_PATH}/MSVD/MSVD_Videos \
    --output_dir ${SAVE_PATH} \
    --lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 300 \
    --datatype msvd \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 2 \
    --tau 5 --K 32 --lambda1 1 --lambda2 20 --lambda3 0.015 \
    --rerank_coe_v 0.05 --rerank_coe_t 0.55 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a log/${job_name}.out
