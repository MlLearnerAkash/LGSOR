import sys
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
import logging
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from mask2former import add_maskformer2_config
import detectron2.utils.comm as comm
from torch.utils.tensorboard import SummaryWriter
import time
import os
from contextlib import contextmanager
import torch
from tqdm import tqdm
import numpy as np
import copy
import cv2
import pickle as pkl
from detectron2.data import build_detection_test_loader
from mask2former.data.dataset_mappers.assr_dataset_mapper import AssrDatasetMapper
from mask2former.data.dataset_mappers.irsr_dataset_mapper import IrsrDatasetMapper
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from pycocotools import mask as coco_mask
import detectron2.utils.comm as comm

from mask2former.data.datasets.register_assr import get_assr_dicts
from mask2former.data.datasets.register_irsr import get_irsr_dicts
from detectron2.data import DatasetFromList, MapDataset

from torch.utils.data import DataLoader,SequentialSampler
import math
logger = logging.getLogger("detectron2")
def find_all_indexes(lst, value):
    indexes = []
    for i in range(len(lst)):
        if lst[i] == value:
            indexes.append(i)
    return indexes
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
@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

def inference(cfg, model,model_name,model_root_dir,datasetmode):
    if comm.is_main_process():
        dataset=cfg.EVALUATION.DATASET
        limited=cfg.EVALUATION.LIMITED
        dataPath=cfg.EVALUATION.DATAPATH

        if dataset=="assr":
            SOR_DATASETPATH=dataPath+"ASSR/"
            print('------Evaluation based on ASSR dataset!------')

            assrdataset = get_assr_dicts(root=SOR_DATASETPATH, mode=datasetmode)
            assrdatasetlist = DatasetFromList(assrdataset, copy=False)
            assrdatasetlist = MapDataset(assrdatasetlist, AssrDatasetMapper(cfg, False))
            dataloader = DataLoader(assrdatasetlist, batch_size=1, shuffle=True, num_workers=0,collate_fn=trivial_batch_collator)

        elif dataset=='irsr':
            SOR_DATASETPATH = dataPath+"IRSR/"
            print('------Evaluation based on IRSR dataset!------')

            irsrdataset = get_irsr_dicts(root=SOR_DATASETPATH, mode=datasetmode)
            irsrdatasetlist = DatasetFromList(irsrdataset, copy=False)
            irsrdatasetlist = MapDataset(irsrdatasetlist, IrsrDatasetMapper(cfg, False))
            dataloader = DataLoader(irsrdatasetlist, batch_size=1, shuffle=False, num_workers=0,collate_fn=trivial_batch_collator)

        ourputdir=f'{cfg.EVALUATION.MODEL_DIR}/{model_name.split(".")[0]}'
        saliencymapPath = os.path.join(ourputdir, 'ResultThres/')
        if not os.path.exists(saliencymapPath):
            os.makedirs(saliencymapPath)

        # colors = [[61, 87, 234], [99, 192, 251], [188, 176, 100], [153, 102, 68], [119, 85, 8]]

        all_time = 0 
        import time 
        with inference_context(model), torch.no_grad():
            res = []
            for idx, inputs in enumerate(dataloader):
                name = inputs[0]["file_name"].split('/')[-1]
                if idx % 100 == 0:
                    print(idx)
                # if os.path.exists(saliencymapPath + '{}.png'.format(name[:-4])):
                #     # print(f'skip {name}')
                #     continue 
                start = time.time()
                predictions=model(inputs)
                all_time += time.time() - start 
                # if idx % 100 == 0:
                #     print(f'speed: {all_time/(idx+1)} s/it')
                img_height = inputs[0]["height"]
                img_width = inputs[0]["width"]
                if "instances" in predictions[-1]:
                    instances = predictions[-1]["instances"].to("cpu")
                    pred_instances = Instances(instances.image_size)
                    flag = False
                    for index in range(len(instances)):
                        score = instances[index].scores
                        if score > cfg.EVALUATION.RESULT_THRESHOLD:
                            if flag == False:
                                pred_instances = instances[index]
                                flag = True
                            else:
                                pred_instances = Instances.cat([pred_instances, instances[index]])
                    gt_masks_polygon = []
                    gt_ranks = []
                    for ins in inputs[0]['annotations']:
                        gt_ranks.append(ins['gt_rank'])
                        segm = []
                        for seg in ins['segmentation']:
                            segm.append(np.asarray(seg))
                        gt_masks_polygon.append(segm)

                    gt_masks = convert_coco_poly_to_mask(gt_masks_polygon, img_height, img_width).cpu().data.numpy()

                    if flag:
                        pred_masks = pred_instances.pred_masks.cpu().data.numpy()
                        pred_ranks = pred_instances.pred_rank.cpu().data.numpy()

                        limited = True 
                        if limited:
                            if dataset == 'assr':
                                if len(pred_ranks) > 5:
                                    sorted_data = sorted(enumerate(pred_ranks), key=lambda x: x[1], reverse=True)
                                    top5_indices = [index for index, value in sorted_data[:5]]
                                    mask = [i in top5_indices for i in range(len(pred_ranks))]

                                    pred_masks = pred_masks[mask, :, :]
                                    pred_ranks = pred_ranks[mask]
                            elif dataset == 'irsr':
                                if len(pred_ranks) > 8:
                                    sorted_data = sorted(enumerate(pred_ranks), key=lambda x: x[1], reverse=True)
                                    top8_indices = [index for index, value in sorted_data[:8]]
                                    mask = [i in top8_indices for i in range(len(pred_ranks))]

                                    pred_masks = pred_masks[mask, :, :]
                                    pred_ranks = pred_ranks[mask]

                        res.append({'gt_masks': [mask for mask in gt_masks], 'segmaps': pred_masks, 'gt_ranks': gt_ranks,'rank_scores': [rank for rank in pred_ranks], 'img_name': name})

                        saliency_rank = [rank for rank in pred_ranks]
                        all_segmaps = np.zeros_like(gt_masks[0], dtype=float)
                        segmaps1 = copy.deepcopy(pred_masks)
                        if len(pred_masks) != 0:
                            color_index = [sorted(saliency_rank).index(a) + 1 for a in saliency_rank]
                            color_len = len(color_index)
                            if dataset == 'assr':
                                if color_len <= 10:
                                    color = [math.floor(255. / 10 * (a + (10 - color_len))) for a in color_index]
                                else:
                                    color = [max(math.floor(255. / 10 * (a + (10 - color_len))), 25) for a in color_index]
                            else:
                                color = [255. / len(saliency_rank) * a for a in color_index]
                            cover_region = all_segmaps != 0

                            for i in range(len(segmaps1), 0, -1):
                                obj_id_list = find_all_indexes(color_index, i)
                                if len(obj_id_list) == 0:
                                    continue
                                else:
                                    for obj_id in obj_id_list:
                                        seg = segmaps1[obj_id]
                                        seg[seg >= 0.5] = color[obj_id]
                                        seg[seg < 0.5] = 0
                                        seg[cover_region] = 0
                                        all_segmaps += seg
                                        cover_region = all_segmaps != 0
                            all_segmaps = all_segmaps.astype(int)

                            cv2.imwrite(saliencymapPath + '{}.png'.format(name[:-4]), all_segmaps)
                    else:
                        all_segmaps = np.zeros_like(gt_masks[0], dtype=int)
                        cv2.imwrite(saliencymapPath + '{}.png'.format(name[:-4]), all_segmaps)
                        segmapsforsasor = np.zeros([0, img_height, img_width])
                        print("Image:"+name+"------Pred_masks is None after confidence threshold")
                        res.append({'gt_masks': [mask for mask in gt_masks], 'segmaps': segmapsforsasor,'gt_ranks': [rank for rank in gt_ranks],'rank_scores': [], 'img_name': name})

            

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    logger.info("Model:\n{}".format(model))

    model_root_dir = cfg.EVALUATION.MODEL_DIR
    datasetmode=cfg.EVALUATION.DATASETMODE
    model_names = cfg.EVALUATION.MODEL_NAMES


    if comm.is_main_process():
        for model_name in model_names:
            model_dir = os.path.join(model_root_dir, model_name)
            checkpoint_logger = logging.getLogger("fvcore.common.checkpoint")
            original_level = checkpoint_logger.level
            checkpoint_logger.setLevel(logging.ERROR)
            
            DetectionCheckpointer(model, save_dir=model_root_dir).resume_or_load(
                model_dir, resume=args.resume
            )
            checkpoint_logger.setLevel(original_level)
            inference(cfg, model,model_name,model_root_dir,datasetmode)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=1,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
