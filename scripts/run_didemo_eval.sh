export CUDA_VISIBLE_DEVICES=9

job_name="ex0"
DATA_PATH="../data/VR_Dataset"
SAVE_PATH="log"
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29555\
    main_pau.py --do_eval --num_thread_reader=8 \
    --epochs=20 --batch_size=32 --n_display=10 \
    --init_model ${MODEL_PATH} \
    --data_path ${DATA_PATH}/DiDeMo/annotation \
    --features_path ${DATA_PATH}/DiDeMo/video \
    --output_dir ${SAVE_PATH} \
    --lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 24 \
    --datatype didemo \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --tau 5 --K 8 --lambda1 1 --lambda2 100 --lambda3 0.025 \
    --rerank_coe_v 0.05 --rerank_coe_t 0.05 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a log/${job_name}.out