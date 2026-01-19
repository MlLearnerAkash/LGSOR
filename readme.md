# Language-Guided Salient Object Ranking

## Model Location

### Trained Model Weights
- **Model File**: `checkpoint/assr_swinl/model.pth`
- **Model File**: `checkpoint/irsr_swinl/model.pth`
- **Config File**: `configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep_assr.yaml`

## Running Inference

### Method 1: Using the Provided Script

The simplest way is to run `demo.sh` directly:

```bash
bash demo.sh
```

### Method 2: Manual Execution

#### 1. Run Inference

**Inference on ASSR dataset:**
```bash
python main.py --num-gpus 1 \
    --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep_assr.yaml \
    EVALUATION.DATASET "assr" \
    EVALUATION.MODEL_DIR "checkpoint/assr_swinl/" \
    EVALUATION.MODEL_NAMES "('model.pth', )"
```


#### 2. Evaluate Results

After inference is complete, run the evaluation script:

```bash
python metric.py --map output/assr_swinl/model/ResultThres
```

## Notes

1. Ensure sufficient GPU memory (recommended >= 16GB)
2. For the first run, CUDA operators need to be compiled:
   ```bash
   cd mask2former/modeling/pixel_decoder/ops
   sh make.sh
   ```
3. If you modify the confidence threshold, the output path will change accordingly


## Citation

```
@inproceedings{liu2025language,
  title={Language-Guided Salient Object Ranking},
  author={Liu, Fang and Liu, Yuhao and Xu, Ke and Ye, Shuquan and Hancke, Gerhard Petrus and Lau, Rynson WH},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={29803--29813},
  year={2025}
}
```

## Contact

For any questions or issues, please contact: **fawnliu2333@gmail.com**