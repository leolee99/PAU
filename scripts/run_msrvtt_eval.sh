export CUDA_VISIBLE_DEVICES=4

job_name="ex0"
DATA_PATH="../data/VR_Dataset"
SAVE_PATH="log"
MODEL_PATH="log/MSRVTT-pytorch_model.bin.0"
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29514\
    main_pau.py --do_eval --num_thread_reader=8 \
    --lr 1e-4 --batch_size=256  --batch_size_val 40 \
    --epochs=5  --n_display=10 \
    --init_model ${MODEL_PATH} \
    --train_csv ${DATA_PATH}/MSRVTT/msrvtt_data/MSRVTT_train.9k.csv \
    --val_csv ${DATA_PATH}/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv \
    --data_path ${DATA_PATH}/MSRVTT/msrvtt_data/MSRVTT_data.json \
    --features_path ${DATA_PATH}/MSRVTT/videos/all \
    --output_dir ${SAVE_PATH} \
    --max_words 32 --max_frames 12 \
    --datatype msrvtt --expand_msrvtt_sentences  \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --tau 5 --K 8 --rerank_coe_v 0.05 --rerank_coe_t 0.05 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a log/${job_name}.out
