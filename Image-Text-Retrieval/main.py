"""
# Pytorch implementation for NeurIPS 2023 paper from
# "arxiv.org/abs/2309.17093."
# "Prototype-based Aleatoric Uncertainty Quantification for Cross-modal Retrieval"
# Built on the top of "https://github.com/leolee99/CLIP_ITM"

# Writen by Hao Li, 2023
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from util import set_seed_logger, get_logger
from params import parse_args
from scheduler import cosine_lr
from eval import evaluate
from modules.model import UCLIP
from modules.criterions import TotalLoss
from dataloader.dataloaders import prepare_coco_dataloaders
from thop import profile

global logger

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def main():
    global logger
    args = parse_args()

    seed = set_seed_logger(args)
    logger = get_logger(os.path.join("log", "log_{}.txt".format(args.experiments)))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model_clip, preprocess = clip.clip.load(args.vision_model, device=device, jit=False) # Must set jit=False for training
    # if device == "cpu":
    #     model_clip.float()
    # else :
    #     clip.model.convert_weights(model_clip) # Actually this line is unnecessary since clip by default already on float16


    if args.resume:
        checkpoint = torch.load(args.resume)
        if 'K_prototype' in checkpoint:
            args.K_prototype = checkpoint['K_prototype']
        model = UCLIP(args, model_clip)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Loaded model from {}".format(args.resume))

    else:
        model = UCLIP(args, model_clip)
        logger.info("Model Initialized!")

    model = model.cuda()

    dataloader = prepare_coco_dataloaders(args, args.dataset_root, preprocess, logger)

    if args.eval:
        train_dataloader = None
        train_length = 0
        batch_num = 0
        args.epochs = 0
        test_dataloader, test_length = dataloader['test']
        Mn_R1 = evaluate(args, model, test_dataloader, logger)
    
    else:
        train_dataloader, train_length = dataloader['train']
        test_dataloader, test_length = dataloader['test']
        batch_num = len(train_dataloader)

    loss_fct = TotalLoss(args, args.tau)
    total_steps = batch_num * args.epochs

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    scaler = GradScaler() if args.precision == "amp" else None

    #Mn_R1 = evaluate(args, model, test_dataloader, logger)

    # add your own code to track the training progress.

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_length)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", total_steps)

    best_score = 0
    #flops, params = profile(model, inputs=(torch.ones(1,3,224,224).long().cuda(), torch.ones(1,77).long().cuda()))
    #print('flops: ', flops, 'params: ', params)
    for epoch in range(args.epochs):
        model.train()

        for idx, batch in enumerate(train_dataloader) :
            step = batch_num * epoch + idx
            scheduler(step)

            optimizer.zero_grad()

            images, texts, _ = batch 
            
            images = images.cuda()
            texts = texts.cuda()

            i2t_sims, t2i_sims, ret = model(images, texts)

            if args.precision == "amp":
                with autocast():
                    total_loss, loss_set = loss_fct(i2t_sims, t2i_sims, ret)
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                scaler.update()
            
            else:
                total_loss, loss_set = loss_fct(i2t_sims, t2i_sims, ret)
                total_loss.backward()
                optimizer.step()


            # if device == "cpu":
            #     optimizer.step()
            # else :
            #     convert_models_to_fp32(model)
            #     optimizer.step()
            #     clip.model.convert_weights(model)

            if (idx % args.display == 0) and (idx != 0):
                logger.info("Epoch: %d/%d, step:%d/%d, lr: %.8f, loss: %f, sim_loss: %f, ucn_loss: %f, Var_loss: %f",
                            epoch + 1, args.epochs, idx, batch_num, optimizer.param_groups[0]['lr'],
                            float(total_loss), float(loss_set['sim_loss']), float(loss_set['uct_loss']), float(loss_set['var_loss']))
        
        file_path = os.path.join(args.checkpoint_path, args.experiments, f"epoch{epoch + 1}.pt")
        save_path = os.path.join(args.checkpoint_path, args.experiments)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(
            {
                "epoch": epoch + 1,
                #"step": steps,
                "name": args.name,
                "K_prototype": args.K_prototype,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            file_path,
        )
        logger.info("Saved checkpoint {} (epoch {})".format(file_path, epoch + 1))

        ## Run on val dataset for selecting best model.
        logger.info("Eval on val dataset")
        Mn_R1 = evaluate(args, model, test_dataloader, logger)

        if best_score <= Mn_R1:
            best_score = Mn_R1
            best_output_model_file = file_path
        logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

if __name__ == '__main__':
    main()

