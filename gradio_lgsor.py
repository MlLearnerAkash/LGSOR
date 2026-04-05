#!/usr/bin/env python3
"""
Gradio interactive visualizer for LGSOR mask ranking.

Features:
  - Browse H5 episodic dataset frames with a slider + Prev/Next buttons.
  - Load a frame → runs model inference → shows side-by-side
      [RGB | GT rank heatmap | Predicted rank heatmap]
    with the original navigation instruction.
  - Edit the instruction (or type a completely custom one) → click
    "Run Custom" → second panel updates with new predictions on the
    same image.  GT panel stays fixed for easy comparison.
  - Spearman ρ and object counts reported for every inference run.

Usage:
  cd costmap_predictor/LGSOR
  python gradio_lgsor.py \
      --h5_path ../langgeonet/subset_10ep.h5 \
      --checkpoint checkpoint/irsr_swinl/model.pth
  # then open  http://localhost:7860
"""

import os
import sys
import argparse
import copy
import io
import textwrap

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import binary_erosion
from scipy.stats import spearmanr
from PIL import Image as PILImage

import gradio as gr

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mask2former import add_maskformer2_config
from h5_lgsor_dataset import H5LGSORDataset

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
_MODEL   = None
_DATASET = None


# ---------------------------------------------------------------------------
# Model / config setup
# ---------------------------------------------------------------------------

def _setup_cfg(config_file, checkpoint_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.EVALUATION.DATASET = "irsr"
    cfg.OUTPUT_DIR = "output/gradio/"
    cfg.freeze()
    return cfg


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _rank_heatmap(masks, ranks, H, W):
    """(H, W) float heatmap; NaN = background.  Most-salient mask wins overlaps."""
    heat = np.full((H, W), np.nan, dtype=np.float32)
    for idx in np.argsort(-np.array(ranks, dtype=float)):   # high → low
        m = np.asarray(masks[idx], dtype=bool)
        heat[m] = float(ranks[idx])
    return heat


def _overlay_masks(ax, masks, ranks, cmap, norm):
    """White contour border + rank integer at centroid for every mask."""
    from matplotlib.colors import to_rgba
    for m, r in zip(masks, ranks):
        m_bool = np.asarray(m, dtype=bool)
        if not m_bool.any():
            continue
        # thin white border (erosion outline)
        border = m_bool & ~binary_erosion(m_bool)
        ys, xs = np.where(border)
        if len(xs):
            ax.scatter(xs, ys, s=0.3, c="white", linewidths=0,
                       alpha=0.55, rasterized=True)
        # rank label at centroid, coloured box matching the heatmap colour
        cy = int(np.where(m_bool)[0].mean())
        cx = int(np.where(m_bool)[1].mean())
        face = to_rgba(cmap(norm(float(r))))
        ax.text(
            cx, cy, str(int(r)),
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=face, alpha=0.9, linewidth=0),
        )


def _draw_panel(ax, rgb_np, masks, ranks, title, cmap, norm, H, W):
    ax.imshow(rgb_np, alpha=0.35)
    if len(masks) == 0:
        ax.set_title(f"{title}\n[no masks]", fontsize=10)
        ax.axis("off")
        return
    masked_heat = np.ma.masked_invalid(_rank_heatmap(masks, ranks, H, W))
    ax.imshow(masked_heat, cmap=cmap, norm=norm,
              alpha=0.65, interpolation="nearest")
    _overlay_masks(ax, masks, ranks, cmap, norm)
    ax.set_title(f"{title}  ({len(masks)} obj)", fontsize=10)
    ax.axis("off")


def _render_figure(rgb_np, gt_masks, gt_ranks, pred_masks, pred_ranks,
                   instruction, spearman_val=None, label_pred="Predicted Ranks"):
    """Return (H, W, 3) uint8 numpy image: [RGB | GT Ranks | Pred Ranks].

    Colormap: RdYlGn_r — green = rank 1 (most salient), red = least salient.
    """
    H, W = rgb_np.shape[:2]

    gt_ranks   = [int(r) for r in gt_ranks]
    pred_ranks = [int(r) for r in pred_ranks]
    gt_masks   = [np.asarray(m, dtype=bool) for m in gt_masks]
    pred_masks = [
        np.asarray(m > 0.5 if m.dtype != bool else m, dtype=bool)
        for m in pred_masks
    ]

    all_r = gt_ranks + pred_ranks
    vmin, vmax = 1, max(max(all_r) if all_r else 1, 2)
    cmap = plt.cm.RdYlGn_r
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=110)
    plt.subplots_adjust(top=0.82, wspace=0.04, left=0.01, right=0.92)

    # Suptitle — wrapped instruction + Spearman
    wrapped = textwrap.fill(instruction, width=95)
    title = f"Instruction: {wrapped}"
    if spearman_val is not None and not np.isnan(spearman_val):
        title += f"\n Spearman ρ = {spearman_val:.3f}"
    fig.suptitle(title, fontsize=9, y=0.98, ha="center", va="top")

    # Panel 0 — RGB
    axes[0].imshow(rgb_np)
    axes[0].set_title("RGB Image", fontsize=10)
    axes[0].axis("off")

    # Panel 1 — GT
    _draw_panel(axes[1], rgb_np, gt_masks,   gt_ranks,   "GT Ranks",
                cmap, norm, H, W)

    # Panel 2 — Predictions
    _draw_panel(axes[2], rgb_np, pred_masks, pred_ranks, label_pred,
                cmap, norm, H, W)

    # Shared colorbar for panels 1 & 2
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.012, pad=0.01)
    cbar.set_label("Rank  (1 = most salient)", fontsize=8)
    ticks = list(range(vmin, vmax + 1))
    if len(ticks) > 10:
        ticks = ticks[::max(1, len(ticks) // 10)]
    cbar.set_ticks(ticks)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(PILImage.open(buf).convert("RGB"))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _gt_from_sample(sample):
    """Extract (gt_masks_np  [K,H,W], gt_ranks [K]) from a dataset sample."""
    gt_masks = sample["instances"].gt_masks
    if hasattr(gt_masks, "tensor"):
        gt_masks_np = gt_masks.tensor.numpy()
    elif isinstance(gt_masks, torch.Tensor):
        gt_masks_np = gt_masks.numpy()
    else:
        gt_masks_np = np.asarray(gt_masks)
    gt_ranks = sample["integer_ranks"].numpy()
    return gt_masks_np, gt_ranks


def _resize_masks(masks_np, rH, rW):
    """Resize (M, h, w) float/bool masks to (M, rH, rW)."""
    if len(masks_np) == 0:
        return masks_np
    mH, mW = masks_np.shape[1], masks_np.shape[2]
    if (mH, mW) == (rH, rW):
        return masks_np
    t = torch.from_numpy(masks_np.astype(np.float32)).unsqueeze(1)
    t = F.interpolate(t, size=(rH, rW), mode="nearest").squeeze(1)
    return t.numpy()


@torch.no_grad()
def _infer(sample, override_instruction=None, conf_thresh=0.3):
    """Run model inference; optionally replace tokens with a new instruction.

    Returns:
        pred_masks_np   (M, H, W) float in [0,1]
        pred_int_ranks  (M,) float integer ranks, 1 = most salient
    """
    s = copy.deepcopy(sample)

    if override_instruction is not None and override_instruction.strip():
        tok = _DATASET._tokenize_text(override_instruction)
        s["tokens"]    = tok["tokens"]
        s["phrases"]   = tok["phrases"]
        s["relations"] = tok["relations"]

    preds = _MODEL([s])
    p = preds[0]

    if "instances" not in p:
        return np.zeros((0, 1, 1), dtype=np.float32), np.array([])

    inst = p["instances"].to("cpu")
    keep = inst.scores > conf_thresh
    if keep.sum() == 0:
        return np.zeros((0, 1, 1), dtype=np.float32), np.array([])

    inst = inst[keep]
    # pred_masks = sigmoid logits (float), pred_rank = raw saliency scores
    pred_masks_np = inst.pred_masks.numpy()         # (M, H, W)  float [0,1]

    try:
        raw_scores = inst.pred_rank.squeeze(-1).numpy()  # (M,)
    except AttributeError:
        raw_scores = inst.scores.numpy()            # fallback: use conf score

    n = len(raw_scores)
    int_ranks = np.zeros(n, dtype=float)
    for pos, i in enumerate(np.argsort(raw_scores)):   # low score = high salience
        int_ranks[i] = pos + 1

    return pred_masks_np, int_ranks


def _quick_spearman(pred_int_ranks, gt_ranks):
    n = min(len(pred_int_ranks), len(gt_ranks))
    if n < 2:
        return float("nan")
    try:
        return float(spearmanr(pred_int_ranks[:n], gt_ranks[:n]).statistic)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Gradio callback functions
# ---------------------------------------------------------------------------

def _sample_and_rgb(idx):
    sample = _DATASET[int(idx)]
    rgb_np = (sample["image"]
              .permute(1, 2, 0)
              .numpy()
              .clip(0, 255)
              .astype(np.uint8))
    return sample, rgb_np


def cb_load(idx, conf_thresh):
    """Load a dataset sample, run inference, return figure + metadata."""
    sample, rgb_np = _sample_and_rgb(idx)
    gt_masks_np, gt_ranks = _gt_from_sample(sample)
    instruction = sample.get("instruction", "")
    rH, rW = rgb_np.shape[:2]

    pred_masks_np, pred_int_ranks = _infer(sample, conf_thresh=conf_thresh)
    pred_masks_np = _resize_masks(pred_masks_np, rH, rW)

    spr = _quick_spearman(pred_int_ranks, gt_ranks)
    vis = _render_figure(
        rgb_np, gt_masks_np, gt_ranks,
        pred_masks_np, pred_int_ranks,
        instruction, spr,
        label_pred="Predicted Ranks (original instr.)",
    )

    n_gt   = len(gt_ranks)
    n_pred = len(pred_int_ranks)
    spr_s  = f"{spr:.3f}" if not np.isnan(spr) else "n/a"
    status = (f"Sample {int(idx)}  |  {sample['file_name']}  |  "
              f"GT objs: {n_gt}  |  Pred objs: {n_pred}  |  "
              f"Spearman ρ = {spr_s}")
    return vis, instruction, status


def cb_prev(idx, conf_thresh):
    new_idx = max(0, int(idx) - 1)
    vis, instr, status = cb_load(new_idx, conf_thresh)
    return new_idx, vis, instr, status


def cb_next(idx, conf_thresh):
    new_idx = min(len(_DATASET) - 1, int(idx) + 1)
    vis, instr, status = cb_load(new_idx, conf_thresh)
    return new_idx, vis, instr, status


def cb_custom(idx, custom_instruction, conf_thresh):
    """Re-run inference with a custom instruction on the same sample."""
    sample, rgb_np = _sample_and_rgb(idx)
    gt_masks_np, gt_ranks = _gt_from_sample(sample)
    rH, rW = rgb_np.shape[:2]

    instr = custom_instruction.strip() or sample.get("instruction", "")

    pred_masks_np, pred_int_ranks = _infer(
        sample, override_instruction=instr, conf_thresh=conf_thresh
    )
    pred_masks_np = _resize_masks(pred_masks_np, rH, rW)

    spr = _quick_spearman(pred_int_ranks, gt_ranks)
    vis = _render_figure(
        rgb_np, gt_masks_np, gt_ranks,
        pred_masks_np, pred_int_ranks,
        instr, spr,
        label_pred="Predicted Ranks (custom instr.)",
    )

    n_gt   = len(gt_ranks)
    n_pred = len(pred_int_ranks)
    spr_s  = f"{spr:.3f}" if not np.isnan(spr) else "n/a"
    metrics = (f"GT objs: {n_gt}  |  Pred objs: {n_pred}  |  "
               f"Spearman ρ = {spr_s}")
    return vis, metrics


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app():
    n = len(_DATASET)
    CSS = """
    #main-img   { border: 2px solid #4a90d9; border-radius: 6px; }
    #custom-img { border: 2px solid #e07b39; border-radius: 6px; }
    """

    with gr.Blocks(title="LGSOR Ranking Visualizer", css=CSS,
                   theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# LGSOR Interactive Ranking Visualizer\n"
            "Browse H5 episodic frames · compare **GT vs predicted** "
            "salience rankings · test **custom navigation instructions** "
            "on the same image."
        )

        # ── Controls row ──────────────────────────────────────────────────
        with gr.Row():
            idx_slider  = gr.Slider(0, n - 1, step=1, value=0,
                                    label=f"Sample index  (0 – {n - 1})",
                                    scale=4)
            conf_slider = gr.Slider(0.0, 1.0, step=0.05, value=0.3,
                                    label="Confidence threshold", scale=1)

        with gr.Row():
            prev_btn = gr.Button("◀  Prev",        scale=1)
            load_btn = gr.Button("Load & Infer",    scale=3, variant="primary")
            next_btn = gr.Button("Next  ▶",        scale=1)

        status_box = gr.Textbox(label="Status", interactive=False)

        # ── Original-instruction result ───────────────────────────────────
        gr.Markdown("### Original instruction result")
        main_img = gr.Image(
            label="RGB  |  GT Ranks  |  Predicted Ranks",
            type="numpy", height=480,
            interactive=False, elem_id="main-img",
        )

        # ── Custom instruction ────────────────────────────────────────────
        gr.Markdown(
            "---\n### Custom instruction\n"
            "Edit the instruction below and click **Run Custom** — the GT "
            "panel stays fixed so you can see exactly what changed."
        )
        with gr.Row():
            instr_box = gr.Textbox(
                label="Navigation instruction (editable)",
                lines=3, scale=5,
            )
            run_btn = gr.Button("Run Custom ▶", variant="secondary", scale=1)

        custom_metrics = gr.Textbox(label="Custom run metrics", interactive=False)
        custom_img = gr.Image(
            label="RGB  |  GT Ranks  |  Predicted Ranks (custom instruction)",
            type="numpy", height=480,
            interactive=False, elem_id="custom-img",
        )

        # ── Legend ────────────────────────────────────────────────────────
        gr.Markdown(
            "**Colormap:** 🟢 green = rank 1 (most salient / closest to goal)  "
            "→  🔴 red = least salient.  "
            "The integer printed on each region is its rank within that panel."
        )

        # ── Wiring ────────────────────────────────────────────────────────
        load_btn.click(
            fn=cb_load,
            inputs=[idx_slider, conf_slider],
            outputs=[main_img, instr_box, status_box],
        )
        prev_btn.click(
            fn=cb_prev,
            inputs=[idx_slider, conf_slider],
            outputs=[idx_slider, main_img, instr_box, status_box],
        )
        next_btn.click(
            fn=cb_next,
            inputs=[idx_slider, conf_slider],
            outputs=[idx_slider, main_img, instr_box, status_box],
        )
        run_btn.click(
            fn=cb_custom,
            inputs=[idx_slider, instr_box, conf_slider],
            outputs=[custom_img, custom_metrics],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _MODEL, _DATASET

    parser = argparse.ArgumentParser(description="LGSOR Gradio Visualizer")
    parser.add_argument("--h5_path",
                        default="../langgeonet/subset_10ep.h5")
    parser.add_argument("--checkpoint",
                        default="checkpoint/irsr_swinl/model.pth")
    parser.add_argument(
        "--config-file",
        default=(
            "configs/coco/instance-segmentation/swin/"
            "maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
        ),
    )
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    # --- Model ---
    print("Loading model …")
    cfg = _setup_cfg(args.config_file, args.checkpoint)
    _MODEL = build_model(cfg)
    DetectionCheckpointer(_MODEL).load(args.checkpoint)
    _MODEL.eval()
    print(f"  Model on {_MODEL.device}  "
          f"({sum(p.numel() for p in _MODEL.parameters()):,} params)")

    # --- Dataset ---
    print("Loading dataset …")
    _DATASET = H5LGSORDataset(h5_path=args.h5_path, image_size=args.image_size)
    print(f"  {len(_DATASET)} samples ready")

    # --- Launch ---
    demo = build_app()
    demo.launch(server_port=args.port, share=args.share, inbrowser=False)


if __name__ == "__main__":
    main()
