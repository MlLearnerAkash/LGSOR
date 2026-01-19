

python main.py --num-gpus 1  --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml  \
    EVALUATION.DATASET "assr" \
    EVALUATION.MODEL_DIR "checkpoint/assr_swinl/" \
    EVALUATION.MODEL_NAMES "('model.pth', )" 

python metric.py --map checkpoint/assr_swinl/model/ResultThres

