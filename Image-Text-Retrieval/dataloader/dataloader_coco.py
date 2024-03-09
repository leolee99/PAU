import os
from torch.utils.data import Dataset
import numpy as np
import clip
from PIL import Image
from pycocotools.coco import COCO
import json
import math

class MSCOCO_Dataset(Dataset):
    def __init__(self, args, image_root, annFile, preprocess, ids=None, subset='train', logger=None):
        logger.info("========== Initializing the %s set ==========", subset)
        self.args = args
        self.image_root = image_root
        self.preprocess = preprocess
        self.subset = subset
        self.num_anns = 5

        self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys()) if ids is None else list(ids)
        self.captions = [self.coco.loadAnns(annotation_id.item())[0]['caption'] for annotation_id in self.ids]
        logger.info('%d captions have been loaded.', len(self.captions))
        self.images_id = [self.coco.loadAnns(annotation_id.item())[0]['image_id'] for annotation_id in self.ids]
        self.image_name = [self.coco.loadImgs(img_id)[0]['file_name'] for img_id in self.images_id]

        self.texts  = clip.clip.tokenize(self.captions)
        self.img_length = len(set(self.images_id))
        self.txt_length = len(self.captions)
        logger.info('%d images have been loaded.', self.img_length)
        logger.info("%s set initialization completed!", subset)
        self.idmap = {}

        # Train with Noise (If you don't want to train with noise, please )
        # ==================================================================================
        if subset == 'train' and args.use_noise:
            self.noise_ratio = args.noise_ratio
            self.noise_num = int(self.noise_ratio * len(self.ids))
            noise_ids = np.random.choice(len(self.ids), self.noise_num, replace=False)
            obj_ids = np.random.choice(noise_ids, self.noise_num, replace=False)
            # for idx, id in enumerate(noise_ids):
            #     self.idmap[id] = obj_ids[idx]
            # np.save('idmap_0.5.npy', self.idmap)
            self.idmap = np.load('noise_map/idmap_{:.1f}.npy'.format(self.noise_ratio), allow_pickle=True).item()
            logger.info("%d noise idx generated done!", self.noise_num)
        # ==================================================================================

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(os.path.join(self.image_root, self.image_name[idx]))) # Image from PIL module
        if idx in self.idmap:
            text = self.texts[self.idmap[idx]]
        else:
            text = self.texts[idx]
        img_id = self.images_id[idx]

        return image, text, img_id