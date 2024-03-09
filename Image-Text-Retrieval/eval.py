import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize

def evaluate(args, model, dataloader, logger, split=1):
    model.eval()
    with torch.no_grad():
        image_features = []
        text_features = []
        num_anns = dataloader.dataset.num_anns
        num_ids = len(dataloader.dataset)
        num_imgs = dataloader.dataset.img_length
        for idx, batch in enumerate(dataloader):

            images, texts, img_id = batch 
            
            images = images.cuda()
            texts = texts.cuda()

            batch_image_features = model.clip.encode_image(images)
            batch_text_features = model.clip.encode_text(texts)

            batch_image_features = batch_image_features / batch_image_features.norm(dim=1, keepdim=True)
            batch_text_features = batch_text_features / batch_text_features.norm(dim=1, keepdim=True)

            image_features.append(batch_image_features)
            text_features.append(batch_text_features)

            if idx % args.display == 0:
                logger.info("step:%d/%d", idx, len(dataloader))

        images_ids = torch.arange(0, num_ids, num_anns).cuda()
        image_features = torch.cat(image_features, dim=0)[images_ids]
        text_features = torch.cat(text_features, dim=0)

        sim_matrix = []
        
        for idx, image_feat in tqdm(enumerate(image_features)):
            logit_scale = model.clip.logit_scale.exp()
            sim_line = logit_scale * image_feat @ text_features.t()

            sim_matrix.append(sim_line.unsqueeze(0).cpu())
        
        sim_matrix = torch.cat(sim_matrix, dim=0)
        label = torch.eye(num_imgs).unsqueeze(-1).repeat(1,1,5).view(-1, num_ids)

        # Rerank
        def objective_function(params):
            beta_1, beta_2 = params
            transformed_matrix = np.exp(-beta_1 * vu.cpu()) * np.exp(-beta_2 * tu.T.cpu()) * sim_matrix

            i2t_r1 = (label==((-transformed_matrix).argsort().argsort() + 1)).sum() / label.shape[0]
            t2i_r1 = (label.T==((-transformed_matrix.T).argsort().argsort() + 1)).sum() / label.shape[1]
    
            count_rank_1 = (i2t_r1 + t2i_r1) / 2

            return -count_rank_1
        
        # learn the best beta
        if args.use_rerank:
            v_alpha, vu, tt_logits = model.image_uncertainty_modeling(image_features)
            t_alpha, tu, vv_logits = model.text_uncertainty_modeling(text_features)

            if args.rerank_learn:
                initial_points = [(x, y) for x in np.arange(args.start_point_range, args.end_point_range, args.step_length) for y in np.arange(args.start_point_range, args.end_point_range, args.step_length)]

                results = []
                for idx, x0 in enumerate(initial_points):
                    results.append(minimize(objective_function, x0, method='BFGS', options={'maxiter': args.max_iterations}))

                best_solution = min(results, key=lambda x: x.fun)
                print(f"Optimized Beta_1: {best_solution.x[0]}")
                print(f"Optimized Beta_2: {best_solution.x[1]}")

                sim_matrix = torch.exp(-best_solution.x[0] * vu.cpu()) * torch.exp(-best_solution.x[1] * tu.T.cpu()) * sim_matrix

            else:
                # simple re-rank using arg parameters directly
                sim_matrix = torch.exp(-args.rk_coe_v * vu.cpu()) * torch.exp(-args.rk_coe_t * tu.T.cpu()) * sim_matrix

        if args.dataset == 'coco':
            # test on 1K
            results = {'i2t_R@1':0, 'i2t_R@5':0, 'i2t_R@10':0, 't2i_R@1':0, 't2i_R@5':0, 't2i_R@10':0, 'mean_R1':0}
            for i in range(5):
                divided_sim = sim_matrix[1000 * i: 1000 * (i + 1) - 1, 5000 * i: 5000 * (i + 1) - 1]
                divided_label = label[1000 * i: 1000 * (i + 1) - 1, 5000 * i: 5000 * (i + 1) - 1]
                result = metric_compute(divided_sim, divided_label, logger)
                for key, values in result.items():
                    results[key] += values
            for key, values in result.items():
                results[key] /= 5

            logger.info("1K Image-to-Text:")
            logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['i2t_R@1'], results['i2t_R@5'], results['i2t_R@10']))
            
            logger.info("1K Text-to-Image:")
            logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['t2i_R@1'], results['t2i_R@5'], results['t2i_R@10']))
            
            logger.info("1K Mean R1: {:.2f}".format(results['mean_R1']))

            # test on COCO 5K
            results = metric_compute(sim_matrix, label, logger)
            
            logger.info("5K Image-to-Text:")
            logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['i2t_R@1'], results['i2t_R@5'], results['i2t_R@10']))
            
            logger.info("5K Text-to-Image:")
            logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['t2i_R@1'], results['t2i_R@5'], results['t2i_R@10']))
            
            logger.info("5K Mean R1: {:.2f}".format(results['mean_R1']))


        elif args.dataset == 'f30k':
            results = metric_compute(sim_matrix, label, logger)
            
            logger.info("Image-to-Text:")
            logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['i2t_R@1'], results['i2t_R@5'], results['i2t_R@10']))
            
            logger.info("Text-to-Image:")
            logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['t2i_R@1'], results['t2i_R@5'], results['t2i_R@10']))
            
            logger.info("Mean R1: {:.2f}".format(results['mean_R1']))
            
    # ground_truth = torch.arange(len(images), dtype=torch.long).cuda()
    return results['mean_R1']


def metric_compute(sim_matrix, label, logger):
    results = {}
    # Image-to-Text
    i2t_rank_matrix = (-sim_matrix).argsort().argsort() + 1
    i2t_gt_rk_matrix = label * i2t_rank_matrix
    i2t_gt_rk_matrix[i2t_gt_rk_matrix==0] = 1e9
    i2t_min_rank = i2t_gt_rk_matrix.min(1).values

    results['i2t_R@1'] = 100 * torch.where(i2t_min_rank <= 1, 1, 0).type(torch.float32).mean()
    results['i2t_R@5'] = 100 * torch.where(i2t_min_rank <= 5, 1, 0).type(torch.float32).mean()
    results['i2t_R@10'] = 100 * torch.where(i2t_min_rank <= 10, 1, 0).type(torch.float32).mean()

    # logger.info("Image-to-Text:")
    # logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
    #             format(results['i2t_R@1'], results['i2t_R@5'], results['i2t_R@10']))
    
    # Text-to-Image
    t2i_rank_matrix = (-sim_matrix.T).argsort().argsort() + 1
    t2i_gt_rk_matrix = label.T * t2i_rank_matrix
    t2i_gt_rk_matrix[t2i_gt_rk_matrix==0] = 1e9
    t2i_min_rank = t2i_gt_rk_matrix.min(1).values

    results['t2i_R@1'] = 100 * torch.where(t2i_min_rank <= 1, 1, 0).type(torch.float32).mean()
    results['t2i_R@5'] = 100 * torch.where(t2i_min_rank <= 5, 1, 0).type(torch.float32).mean()
    results['t2i_R@10'] = 100 * torch.where(t2i_min_rank <= 10, 1, 0).type(torch.float32).mean()

    # logger.info("Text-to-Image:")
    # logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
    #             format(results['t2i_R@1'], results['t2i_R@5'], results['t2i_R@10']))
    
    results['mean_R1'] = (results['i2t_R@1'] + results['t2i_R@1']) / 2

    # logger.info("Mean R1: {:.2f}".format(results['mean_R1']))
    
    return results



    