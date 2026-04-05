"""
Evaluate the LGSOR irsr_swinl model on H5 episodic data (subset_10ep.h5).

The H5 dataset contains per-episode frames with pre-computed object masks
and geodesic cost rankings. This script:
  1. Builds the LGSOR MaskFormer model from the swin-large config
  2. Loads the irsr_swinl checkpoint
  3. Creates a dataloader from the H5 file using H5LGSORDataset
  4. Runs inference and evaluates saliency ranking predictions
     against the ground-truth geodesic-cost-based rankings.

Usage:
  cd costmap_predictor/LGSOR
  python eval_h5.py --h5_path ../langgeonet/subset_10ep.h5 \
      --checkpoint checkpoint/irsr_swinl/model.pth \
      --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml
"""

import os
import sys
import argparse
import logging
import math
import copy
from collections import defaultdict

import textwrap

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, to_rgba
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Instances, BitMasks, ImageList

# LGSOR imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mask2former import add_maskformer2_config
from h5_lgsor_dataset import H5LGSORDataset, lgsor_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _build_rank_composite(masks, ranks, H, W):
    """Combine per-object masks into a single rank heatmap [H, W].

    Each pixel gets the rank of the mask it belongs to.  On overlaps the
    more salient (lower rank) mask wins.
    """
    heatmap = np.full((H, W), np.nan, dtype=np.float32)
    # Process in descending rank order so the most salient (rank=1) writes last
    order = np.argsort(-np.array(ranks, dtype=float))  # high rank → low rank
    for idx in order:
        m = np.asarray(masks[idx], dtype=bool)
        heatmap[m] = float(ranks[idx])
    return heatmap


def _mask_contour_overlay(ax, masks, ranks, cmap, norm):
    """Draw white contour outlines + rank label at centroid for each mask."""
    from matplotlib.colors import to_rgba
    for m, r in zip(masks, ranks):
        m_bool = np.asarray(m, dtype=bool)
        if m_bool.sum() == 0:
            continue
        # Contour
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(m_bool)
        outline = m_bool & ~eroded
        ys, xs = np.where(outline)
        if len(xs):
            ax.scatter(xs, ys, s=0.3, c="white", linewidths=0, alpha=0.6, rasterized=True)
        # Centroid label
        ys_m, xs_m = np.where(m_bool)
        cy, cx = int(ys_m.mean()), int(xs_m.mean())
        face = to_rgba(cmap(norm(float(r))))
        ax.text(
            cx, cy, str(int(r)),
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=face, alpha=0.85, linewidth=0),
        )


def visualize_ranking(
    rgb_np,          # (H, W, 3) uint8
    gt_masks,        # list/array of (H, W) bool arrays
    gt_ranks,        # list/array of int, rank 1 = most salient
    pred_masks,      # list/array of (H, W) bool/float arrays
    pred_ranks,      # list/array of int or float ranks
    instruction,     # str
    save_path,       # output .png path
):
    """Save a 3-panel figure: RGB | GT rank heatmap | Predicted rank heatmap.

    Each mask region is filled with a colour keyed to its rank
    (RdYlGn colormap: green=rank-1/most-salient, red=least-salient).
    The integer rank is printed at each mask centroid.
    """
    from scipy.ndimage import binary_erosion  # noqa – just ensure import here

    H, W = rgb_np.shape[:2]
    gt_ranks  = [int(r) for r in gt_ranks]
    pred_ranks = [int(r) for r in pred_ranks]
    gt_masks  = [np.asarray(m, dtype=bool) for m in gt_masks]
    pred_masks = [np.asarray(m > 0.5 if m.dtype != bool else m, dtype=bool)
                  for m in pred_masks]

    all_ranks  = gt_ranks + pred_ranks
    vmin, vmax = 1, max(max(all_ranks) if all_ranks else 1, 2)

    cmap = plt.cm.RdYlGn_r   # rank-1 → green, high-rank → red
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=120)
    plt.subplots_adjust(top=0.84, wspace=0.05)

    # ── Title ────────────────────────────────────────────────────────────────
    wrapped = textwrap.fill(instruction, width=90)
    fig.suptitle(f"Instruction:\n{wrapped}", fontsize=10, y=0.98,
                 ha="center", va="top")

    # ── Panel 0: RGB image ───────────────────────────────────────────────────
    axes[0].imshow(rgb_np)
    axes[0].set_title("RGB Image", fontsize=11)
    axes[0].axis("off")

    # ── Panel helper ─────────────────────────────────────────────────────────
    def draw_panel(ax, masks, ranks, title):
        # Faded background
        ax.imshow(rgb_np, alpha=0.35)
        if len(masks) == 0 or len(ranks) == 0:
            ax.set_title(title + "  [no masks]", fontsize=11)
            ax.axis("off")
            return
        heat = _build_rank_composite(masks, ranks, H, W)
        # Mask NaN pixels (background)
        masked_heat = np.ma.masked_invalid(heat)
        ax.imshow(masked_heat, cmap=cmap, norm=norm, alpha=0.65, interpolation="nearest")
        _mask_contour_overlay(ax, masks, ranks, cmap, norm)
        ax.set_title(f"{title}  ({len(masks)} objects)", fontsize=11)
        ax.axis("off")

    # ── Panel 1: GT ──────────────────────────────────────────────────────────
    draw_panel(axes[1], gt_masks, gt_ranks, "GT Ranks")

    # ── Panel 2: Predictions ─────────────────────────────────────────────────
    draw_panel(axes[2], pred_masks, pred_ranks, "Predicted Ranks")

    # ── Shared colorbar ──────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1:], fraction=0.015, pad=0.01)
    cbar.set_label("Rank  (1 = most salient)", fontsize=9)
    tick_vals = list(range(vmin, vmax + 1))
    if len(tick_vals) > 10:
        tick_vals = tick_vals[::max(1, len(tick_vals) // 10)]
    cbar.set_ticks(tick_vals)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def setup_cfg(config_file, checkpoint_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    # Override weights to point to checkpoint (used by build_model init, but
    # DetectionCheckpointer will overwrite with the actual checkpoint)
    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.EVALUATION.DATASET = "irsr"
    cfg.OUTPUT_DIR = "output/h5_eval/"
    cfg.freeze()
    return cfg


def compute_ranking_metrics(pred_ranks, gt_ranks):
    """Compute ranking correlation metrics between predicted and GT ranks."""
    if len(pred_ranks) < 2:
        return {"spearman": float("nan"), "kendall": float("nan"), "top1_acc": float("nan")}

    pred_ranks = np.array(pred_ranks, dtype=float)
    gt_ranks = np.array(gt_ranks, dtype=float)

    try:
        spr = spearmanr(pred_ranks, gt_ranks).statistic
    except Exception:
        try:
            spr = spearmanr(pred_ranks, gt_ranks).correlation
        except Exception:
            spr = float("nan")

    try:
        kt = kendalltau(pred_ranks, gt_ranks).statistic
    except Exception:
        try:
            kt = kendalltau(pred_ranks, gt_ranks).correlation
        except Exception:
            kt = float("nan")

    # Top-1 accuracy: does the highest-ranked prediction match GT?
    pred_top1 = np.argmax(pred_ranks)
    gt_top1 = np.argmax(gt_ranks)
    top1_acc = 1.0 if pred_top1 == gt_top1 else 0.0

    return {"spearman": spr, "kendall": kt, "top1_acc": top1_acc}


def iou_matrix(pred_masks, gt_masks):
    """Compute IoU between each pair of pred and GT masks.
    pred_masks: [M, H, W] binary
    gt_masks:   [N, H, W] binary
    Returns: [M, N] IoU matrix
    """
    M = pred_masks.shape[0]
    N = gt_masks.shape[0]
    pred_flat = pred_masks.reshape(M, -1).float()
    gt_flat = gt_masks.reshape(N, -1).float()
    intersection = torch.mm(pred_flat, gt_flat.t())
    pred_area = pred_flat.sum(dim=1, keepdim=True)
    gt_area = gt_flat.sum(dim=1, keepdim=True)
    union = pred_area + gt_area.t() - intersection
    return intersection / (union + 1e-6)


def match_predictions_to_gt(pred_masks, pred_ranks, gt_masks, gt_ranks, iou_thresh=0.5):
    """Match predicted instances to GT instances using IoU, then compare ranks.

    Returns matched (pred_rank, gt_rank) pairs.
    """
    if pred_masks.shape[0] == 0 or gt_masks.shape[0] == 0:
        return [], []

    ious = iou_matrix(pred_masks, gt_masks)  # [M, N]

    matched_pred_ranks = []
    matched_gt_ranks = []
    used_gt = set()

    # Greedy matching: for each pred, find best GT match
    for i in range(pred_masks.shape[0]):
        best_j = -1
        best_iou = iou_thresh
        for j in range(gt_masks.shape[0]):
            if j in used_gt:
                continue
            if ious[i, j] > best_iou:
                best_iou = ious[i, j].item()
                best_j = j
        if best_j >= 0:
            used_gt.add(best_j)
            matched_pred_ranks.append(pred_ranks[i])
            matched_gt_ranks.append(gt_ranks[best_j])

    return matched_pred_ranks, matched_gt_ranks


@torch.no_grad()
def evaluate(model, dataloader, device, confidence_threshold=0.3,
             vis_dir=None, vis_max=50):
    """Run inference and evaluate ranking predictions."""
    model.eval()
    vis_count = 0

    all_results = []
    all_spearman = []
    all_kendall = []
    all_top1 = []

    # Also track direct ranking (without mask matching)
    all_direct_spearman = []
    all_direct_kendall = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        predictions = model(batch)

        for sample, pred in zip(batch, predictions):
            gt_ranks = sample["integer_ranks"].numpy()
            gt_masks = sample["instances"].gt_masks  # [K, H, W]
            K_gt = len(gt_ranks)
            file_name = sample["file_name"]
            instruction = sample.get("instruction", "")

            if "instances" not in pred:
                logger.debug(f"{file_name}: no instances predicted")
                continue

            instances = pred["instances"].to("cpu")

            # Filter by confidence
            keep = instances.scores > confidence_threshold
            if keep.sum() == 0:
                logger.debug(f"{file_name}: no instances above threshold")
                continue

            instances = instances[keep]
            pred_masks = (instances.pred_masks > 0.5).float()
            pred_rank_scores = instances.pred_rank.numpy()

            # Convert pred rank scores to integer ranks (higher score = higher rank)
            n_pred = len(pred_rank_scores)
            sorted_idx = np.argsort(pred_rank_scores)
            pred_integer_ranks = np.zeros(n_pred, dtype=float)
            for rank_pos, obj_idx in enumerate(sorted_idx):
                pred_integer_ranks[obj_idx] = rank_pos + 1

            # Resize GT masks to match prediction size if needed
            pred_h, pred_w = pred_masks.shape[1], pred_masks.shape[2]
            gt_h, gt_w = gt_masks.shape[1], gt_masks.shape[2]
            if (pred_h, pred_w) != (gt_h, gt_w):
                gt_masks_resized = F.interpolate(
                    gt_masks.float().unsqueeze(1),
                    size=(pred_h, pred_w),
                    mode="nearest"
                ).squeeze(1).bool()
            else:
                gt_masks_resized = gt_masks

            # Match predictions to GT via IoU
            matched_pred, matched_gt = match_predictions_to_gt(
                pred_masks, pred_integer_ranks,
                gt_masks_resized.float(), gt_ranks.astype(float),
                iou_thresh=0.5
            )

            if len(matched_pred) >= 2:
                metrics = compute_ranking_metrics(matched_pred, matched_gt)
                all_spearman.append(metrics["spearman"])
                all_kendall.append(metrics["kendall"])
                all_top1.append(metrics["top1_acc"])
            elif len(matched_pred) == 1:
                all_top1.append(1.0 if matched_pred[0] == matched_gt[0] else 0.0)

            # Direct ranking eval: use the model's own ranking order
            # (no mask matching, just compare rank distributions)
            if n_pred >= 2 and K_gt >= 2:
                # Take top-K predictions where K = number of GT objects
                if n_pred > K_gt:
                    top_k_idx = np.argsort(-pred_rank_scores)[:K_gt]
                    top_k_scores = pred_rank_scores[top_k_idx]
                else:
                    top_k_scores = pred_rank_scores

                # Rank correlation between top predicted scores and GT ranks
                sorted_pred = np.argsort(-top_k_scores)  # descending order of pred scores
                sorted_gt = np.argsort(-gt_ranks[:len(top_k_scores)])  # descending order of GT

                if len(sorted_pred) >= 2:
                    try:
                        dspr = spearmanr(
                            np.arange(len(sorted_pred)),
                            [np.where(sorted_gt == i)[0][0] for i in sorted_pred]
                        ).statistic
                    except Exception:
                        dspr = float("nan")
                    all_direct_spearman.append(dspr)

            all_results.append({
                "file_name": file_name,
                "n_gt": K_gt,
                "n_pred": n_pred,
                "n_matched": len(matched_pred),
                "gt_ranks": gt_ranks.tolist(),
                "pred_rank_scores": pred_rank_scores.tolist(),
            })

            # ── Visualization ───────────────────────────────────────────────
            if vis_dir is not None and vis_count < vis_max:
                # RGB image: (3,H,W) float → (H,W,3) uint8
                img_t = sample["image"]       # (3,H,W) float, range [0,255]
                rgb_np = img_t.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)

                # GT masks and ranks
                if hasattr(gt_masks, "tensor"):
                    gt_masks_np = gt_masks.tensor.numpy()   # (K,H,W)
                else:
                    gt_masks_np = gt_masks.numpy()           # (K,H,W)

                # Pred masks and integer ranks after filtering
                pred_masks_np = pred_masks.numpy()           # (M,H,W)
                # Resize pred masks to match RGB if needed
                pH, pW = pred_masks_np.shape[1], pred_masks_np.shape[2]
                rH, rW = rgb_np.shape[:2]
                if (pH, pW) != (rH, rW):
                    pred_masks_t = torch.from_numpy(pred_masks_np.astype(np.float32)).unsqueeze(1)
                    pred_masks_np = F.interpolate(
                        pred_masks_t, size=(rH, rW), mode="nearest"
                    ).squeeze(1).numpy()

                safe_name = file_name.replace("/", "_").replace(" ", "_")
                vis_path = os.path.join(vis_dir, f"{safe_name}.png")
                visualize_ranking(
                    rgb_np,
                    gt_masks_np, gt_ranks,
                    pred_masks_np, pred_integer_ranks,
                    instruction,
                    vis_path,
                )
                vis_count += 1

    # Aggregate
    def safe_mean(arr):
        arr = [x for x in arr if not np.isnan(x)]
        return np.mean(arr) if arr else float("nan")

    report = {
        "n_samples": len(all_results),
        "n_with_matches": sum(1 for r in all_results if r["n_matched"] > 0),
        "avg_n_pred": safe_mean([r["n_pred"] for r in all_results]),
        "avg_n_gt": safe_mean([r["n_gt"] for r in all_results]),
        "avg_n_matched": safe_mean([r["n_matched"] for r in all_results]),
        "spearman_rank_corr (mask-matched)": safe_mean(all_spearman),
        "kendall_tau (mask-matched)": safe_mean(all_kendall),
        "top1_accuracy (mask-matched)": safe_mean(all_top1),
        "spearman_rank_corr (direct)": safe_mean(all_direct_spearman),
    }
    return report, all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LGSOR on H5 data")
    parser.add_argument("--h5_path", type=str,
                        default="../langgeonet/subset_10ep.h5",
                        help="Path to the H5 file")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoint/irsr_swinl/model.pth",
                        help="Path to the model checkpoint")
    parser.add_argument("--config-file", type=str,
                        default="configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
                        help="Path to the config file")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples for quick testing")
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="Directory to save ranking heatmap visualizations. "
                             "Skipped if not set.")
    parser.add_argument("--vis_max", type=int, default=50,
                        help="Maximum number of visualizations to save (default: 50)")
    args = parser.parse_args()

    # Setup config and model
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"H5 file: {args.h5_path}")

    cfg = setup_cfg(args.config_file, args.checkpoint)
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Load checkpoint
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(args.checkpoint)
    logger.info("Checkpoint loaded successfully")

    device = model.device

    # Create dataset
    dataset = H5LGSORDataset(
        h5_path=args.h5_path,
        image_size=args.image_size,
    )

    if args.max_samples and args.max_samples < len(dataset):
        dataset.samples = dataset.samples[:args.max_samples]
        logger.info(f"Limiting to {args.max_samples} samples")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lgsor_collate_fn,
    )

    logger.info(f"Dataset: {len(dataset)} samples")
    logger.info(f"Running evaluation with confidence_threshold={args.confidence_threshold}")
    if args.vis_dir:
        logger.info(f"Visualizations will be saved to {args.vis_dir} (max {args.vis_max})")

    # Evaluate
    report, results = evaluate(model, dataloader, device,
                               confidence_threshold=args.confidence_threshold,
                               vis_dir=args.vis_dir,
                               vis_max=args.vis_max)

    # Print report
    print("\n" + "=" * 60)
    print("LGSOR Evaluation Report (irsr_swinl on H5 data)")
    print("=" * 60)
    for key, val in report.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    print("=" * 60)

    # Save results
    os.makedirs("output/h5_eval", exist_ok=True)
    import json
    with open("output/h5_eval/eval_results.json", "w") as f:
        json.dump({"report": report, "per_sample": results[:50]}, f, indent=2, default=str)
    logger.info("Results saved to output/h5_eval/eval_results.json")


if __name__ == "__main__":
    main()
