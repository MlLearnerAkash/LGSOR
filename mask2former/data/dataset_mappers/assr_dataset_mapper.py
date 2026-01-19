# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
# from . import detection_utils as utils
# from . import transforms as T
from pycocotools import mask as coco_mask
"""
This file contains the default mapping that's applied to "dataset dicts".
"""
import json 

__all__ = ["AssrDatasetMapper"]
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    #assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    augmentation.extend([
        T.Resize((image_size, image_size)),
    ])


    return augmentation
import os 
from transformers import BertTokenizer
class AssrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            size_divisibility: pad image size to be divisible by this value
        """

        # fmt: off
        self.is_train = is_train
        self.image_format = image_format
        self.tfm_gens = tfm_gens
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logging.getLogger(__name__).info(
            "[ASSRdatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        ## 
        self.summarize = 'llava' 
        cap_path = 'data/summarizes_all.json'
        coco_caps = json.load(open(cap_path, 'r'))
        phrase_path = 'data/nouns_llava.json'
        
        rel_path = 'data/relation_llava.json'
        if phrase_path is not None:
            phrase_info = json.load(open(phrase_path, 'r'))
        if rel_path is not None:
            relation_info = json.load(open(rel_path, 'r'))

        # print('cap_path', cap_path, '---', f'summarize: {summarize}')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_tokens = 256 

        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.relation_info = relation_info
        self.phrase_info = phrase_info
        self.coco_caps = coco_caps
        self.phrase_path = phrase_path
        self.rel_path = rel_path


    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        ######
        phrase_seq_len, num_phrases = 8+2, 25  ## +2 for cls and sep
        num_relations = 18
        my_img_id = os.path.basename(dataset_dict["file_name"]).split('.')[0]
        # print(my_img_id, '======')
        captions = [self.coco_caps[my_img_id]]
        dataset_dict['captions'] = captions
        tokens = self.tokenizer(
            captions, padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors='pt'
        )
        dataset_dict['tokens'] = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
        if self.phrase_path is not None:
            if 'coco' in self.summarize:
                this_phrase_info = self.phrase_info[my_img_id][coco_select_index]
            else:
                this_phrase_info = self.phrase_info[my_img_id]
                
            # if 'random' in summarize:
            #     random.shuffle(this_phrase_info)
            phrases = [item[0] for item in this_phrase_info]
            p_pos_l = [item[1] for item in this_phrase_info]
            p_pos_r = [item[2] for item in this_phrase_info]

            phrase_masks = []
            tokenized_phrases = []
            for phrase, pos_l, pos_r in zip(phrases, p_pos_l, p_pos_r):
                assert pos_l < pos_r
                tokenized_phrase = self.tokenizer(
                    phrase,
                    padding='max_length',
                    max_length=phrase_seq_len,
                    truncation=True,
                    return_tensors='pt',
                )
                tokenized_phrases.append(tokenized_phrase['input_ids'][0])  
                phrase_masks.append(tokenized_phrase['attention_mask'][0])
            
            for _ in range(len(phrases), num_phrases):
                tokenized_phrase = self.tokenizer(
                    "",
                    padding='max_length',
                    max_length=phrase_seq_len,
                    return_tensors='pt',
                )
                ## default token include [CLS] and [SEP], even no phrase
                tokenized_phrases.append(tokenized_phrase['input_ids'][0])
                phrase_masks.append(tokenized_phrase['attention_mask'][0])
                p_pos_l.append(0)
                p_pos_r.append(1)
            p_in_sent_mask = []
            for j in range(num_phrases):
                mask = torch.ones_like(dataset_dict['tokens']['attention_mask'][0, :])
                mask[p_pos_l[j]:p_pos_r[j]] = 0
                p_in_sent_mask.append(mask)
            dataset_dict['phrases'] = {
                'p_input_ids': torch.stack(tokenized_phrases),
                'p_attention_mask': torch.stack(phrase_masks),
                # 'pos_l': torch.stack(p_pos_l),
                # 'pos_r': torch.stack(p_pos_r),
                'p_in_sent_mask': torch.stack(p_in_sent_mask),
            }
        else:
            dataset_dict['phrases'] = None
        if self.rel_path is not None:
            this_rel_info = self.relation_info[my_img_id]
            relations = [item[0] for item in this_rel_info]
            r_pos_l = [item[1] for item in this_rel_info]
            r_pos_r = [item[2] for item in this_rel_info]
            # ------
            relation_masks = []
            tokenized_relations = []
            for relation, pos_l, pos_r in zip(relations, r_pos_l, r_pos_r):
                assert pos_l < pos_r
                tokenized_relation = self.tokenizer(
                    relation,
                    padding='max_length',
                    max_length=phrase_seq_len,
                    truncation=True,
                    return_tensors='pt',
                )
                tokenized_relations.append(tokenized_relation['input_ids'][0])  
                relation_masks.append(tokenized_relation['attention_mask'][0])
            
            for _ in range(len(relations), num_relations):
                tokenized_relation = self.tokenizer(
                    "",
                    padding='max_length',
                    max_length=phrase_seq_len,
                    return_tensors='pt',
                )
                ## default token include [CLS] and [SEP], even no relation
                tokenized_relations.append(tokenized_relation['input_ids'][0])
                relation_masks.append(tokenized_relation['attention_mask'][0])
                r_pos_l.append(0)
                r_pos_r.append(1)
            r_in_sent_mask = []
            for j in range(num_relations):
                mask = torch.ones_like(dataset_dict['tokens']['attention_mask'][0, :])
                mask[r_pos_l[j]:r_pos_r[j]] = 0
                r_in_sent_mask.append(mask)
            dataset_dict['relations'] = {
                'r_input_ids': torch.stack(tokenized_relations),
                'r_attention_mask': torch.stack(relation_masks),
                # 'pos_l': torch.stack(r_pos_l),
                # 'pos_r': torch.stack(r_pos_r),
                'r_in_sent_mask': torch.stack(r_in_sent_mask),
            }
        else:
            dataset_dict['relations'] = None
        ######


        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            #dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            gt_ranks = []
            for anno in dataset_dict["annotations"]:
                gt_ranks.append(anno['gt_rank'])
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.

            #instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances)


            h, w = instances.image_size

            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            dataset_dict["instances"] = instances
            dataset_dict["instances"].gt_ranks = torch.LongTensor(gt_ranks)

        return dataset_dict

