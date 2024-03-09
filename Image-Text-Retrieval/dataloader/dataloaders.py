import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader.dataloader_coco import MSCOCO_Dataset

def _get_coco_file_paths(dataset_root):
    """Select proper train / val classes and omit id files.
    """
    train_ids = np.load(os.path.join(dataset_root,'annotations/coco_train_ids.npy'))
    train_extra_ids = np.load(os.path.join(dataset_root,'annotations/coco_restval_ids.npy'))
    val_ids = np.load(os.path.join(dataset_root,'annotations/coco_dev_ids.npy'))[:5000]
    te_ids = np.load(os.path.join(dataset_root,'annotations/coco_test_ids.npy'))

    image_root = os.path.join(dataset_root, 'images/')
    train_ann = os.path.join(dataset_root, 'annotations/captions_train2014.json')
    val_ann = os.path.join(dataset_root, 'annotations/captions_val2014.json')

    return train_ids, train_extra_ids, val_ids, te_ids, image_root, train_ann, val_ann

def dataloader_mscoco_train(args, image_root, annFile, preprocess, ids, subset, logger):
    msrvtt_dataset = MSCOCO_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    #train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        #shuffle=(train_sampler is None),
        #sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset)#, train_sampler

def dataloader_mscoco_test(args, image_root, annFile, preprocess, ids, subset, logger):
    msrvtt_dataset = MSCOCO_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.eval_batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset)#, train_sampler

def prepare_coco_dataloaders(args,
                             dataset_root,
                             preprocess,
                             logger,):
    """Prepare MS-COCO Caption train / val / test dataloaders
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/coco_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 6)
    Returns:
        dataloaders (dict): keys = ["train", "val", "te"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """

    train_ids, train_extra_ids, val_ids, test_ids, image_root, train_ann, val_ann = _get_coco_file_paths(dataset_root)

    dataloaders = {}

    if args.eval:
        dataloaders['train'] = None, None
    else:
        dataloaders['train'] = dataloader_mscoco_train(
            args, image_root, train_ann, preprocess, 
            train_ids, 'train', logger,
        )

    # dataloaders['val'] = dataloader_mscoco_val(
    #     image_root, val_ann, val_ids, vocab,
    #     num_workers=num_workers, batch_size=eval_batch_size,
    #     train=False, cxc_path=cxc_val_path,
    #     tokenizer=tokenizer,
    # )

    dataloaders['test'] = dataloader_mscoco_test(
        args, image_root, val_ann, preprocess, 
        test_ids, 'test', logger,
    )

    return dataloaders