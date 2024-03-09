import argparse

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN50", "RN101", "RN50x4"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name in ["ViT-B-32", "ViT-B-16", "ViT-H-14"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    elif model_name in ["ViT-L-14", "ViT-L-14-336"]:
        return {"lr": 4.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        type=str,
        default='ex0',
        help="Experiments Name",
    )
    parser.add_argument(
        "--dataset",
        choices=["coco", "f30k"],
        default="coco",
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default='/home/lihao/data/MSCOCO',
        help="Path to the dataset",
    )
    parser.add_argument(
        "--use_noise", action="store_true", default=False, help="wether to train with noise."
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=0.2, help="The ratio of noise."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of full epochs to train for (only works if --max_steps is None)."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="train or eval",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=64, help="Batch size for eval per GPU."
    )
    parser.add_argument(
        "--display", type=int, default=50, help="The steps interval of display."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default='/home/lihao/data/CLIP/log',
        help="Path to save the model",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="The number of workers for dataloader."
    )        
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="pau",
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--context_length", type=int, default=64, help="The maximum length of input text (include [CLS] & [SEP] tokens)."
    )
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon.")
    parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=500, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        '--max_iterations', type=int, default=3, help='Max interation times while learning best beta parameters in the rerank process.'
    ) 
    parser.add_argument(
        '--start_point_range', type=float, default=0.01, help='The start of the range in the rerank learning.'
    )   
    parser.add_argument(
        '--end_point_range', type=float, default=1.01, help='The end of the range in the rerank learning.'
    )
    parser.add_argument(
        '--step_length', type=float, default=0.05, help='The step of each point gap in the rerank learning.'
        )
    parser.add_argument(
        "--use_rerank", action="store_true", default=False, help="wether to employ rerank."
    )
    parser.add_argument(
        "--rerank_learn", action="store_true", default=False, help="wether to learn best rerank coefficient."
    )        
    parser.add_argument(
        "--rk_coe_v", type=float, default=0.05, help="the weight of vision modality while rerank."
    )
    parser.add_argument(
        "--rk_coe_t", type=float, default=0.05, help="the weight of textual modality while rerank."
    )    
    parser.add_argument(
        "--K_prototype", type=int, default=8, help="The number of prototypes."
    )
    parser.add_argument(
        "--tau", type=int, default=5, help="The parameter of ucn loss."
    )
    parser.add_argument(
        "--uct_weight", type=float, default=0.5, help="The weight of uct loss."
    )
    parser.add_argument(
        "--var_weight", type=float, default=0.015, help="The weight of var loss."
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument(
        "--vision_model",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px", "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"],
        default="ViT-B/32",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--freeze_vision",
        action="store_true",
        default=False,
        help="Freeze the weight of vision encoder.",
    )
    # arguments for distributed training
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=123, 
        help="Random seed."
    )
    args = parser.parse_args()

    return args
