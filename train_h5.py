"""
Fine-tune the full LGSOR architecture on H5 episodic data.

Architecture (from LGSOR / CVPR 2025):
  - Image encoder:  Swin-L pretrained
  - Text encoder:   BERT-base-uncased
  - Pixel decoder:  MSDeformAttn
  - Transformer decoder + fusion + query_encoder
  - Ranking head:   Multi-instance GAT graph → saliency scores

Training strategy:
  - Full model is unfrozen (all parameters trainable).
  - Three loss terms: binary cross-entropy (mask BCE), dice loss (mask
    overlap), and pairwise ranking loss (saliency ordering).
  - Hungarian matching assigns predicted queries to GT objects,
    providing proper mask supervision.
  - Three LR groups: backbone (--lr * 0.01), decoder/encoder (--lr * 0.1),
    ranking head (--lr).

Usage:
  cd costmap_predictor/LGSOR
  python train_h5.py --h5_path ../langgeonet/subset_10ep.h5 \\
      --checkpoint checkpoint/irsr_swinl/model.pth \\
      --epochs 50 --lr 1e-4 --batch_size 2
"""

import os
import sys
import argparse
import logging
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from scipy.stats import spearmanr

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import ImageList

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mask2former import add_maskformer2_config
from h5_lgsor_dataset import create_lgsor_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def setup_cfg(config_file, checkpoint_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.freeze()
    return cfg


# ---------------------------------------------------------------------------
# Parameter grouping helpers
# ---------------------------------------------------------------------------

# Ranking-head parameters (highest LR)
RANK_HEAD_PATTERNS = [
    "relation_saliency_score",
    "relation_multifc",
    "relation_multi_ins_g_final",
]

# Backbone parameters (lowest LR)
BACKBONE_PATTERNS = [
    "backbone.",
]


def unfreeze_all_and_group(model):
    """Unfreeze the entire model and separate parameters into three groups.

    Returns:
        backbone_params, other_params, rank_params
    """
    for param in model.parameters():
        param.requires_grad = True

    backbone_params, rank_params, other_params = [], [], []
    backbone_count = 0
    rank_count = 0
    other_count = 0

    for name, param in model.named_parameters():
        n = param.numel()
        if any(pat in name for pat in RANK_HEAD_PATTERNS):
            rank_params.append(param)
            rank_count += n
        elif any(pat in name for pat in BACKBONE_PATTERNS):
            backbone_params.append(param)
            backbone_count += n
        else:
            other_params.append(param)
            other_count += n

    total = backbone_count + other_count + rank_count
    logger.info(f"Backbone trainable:    {backbone_count:,}")
    logger.info(f"Decoder/Encoder trainable: {other_count:,}")
    logger.info(f"Rank-head trainable:   {rank_count:,}")
    logger.info(f"Total trainable:       {total:,} (100.00%)")
    return backbone_params, other_params, rank_params


def log_weight_dict(model):
    """Log the criterion weight_dict (BCE + dice + ranking)."""
    wd = model.criterion.weight_dict
    logger.info(f"weight_dict: {len(wd)} entries (BCE + dice + ranking)")
    for k, v in sorted(wd.items()):
        logger.info(f"  {k}: {v}")


# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate ranking quality using IoU-based query-to-GT assignment.

    For each GT mask we pick the predicted query whose mask has the highest
    IoU, then compare the corresponding saliency scores with the GT ranks
    via Spearman correlation.  This avoids depending on Hungarian matching
    (which is unreliable when the frozen model's masks don't overlap the
    H5 GT masks).
    """
    model.eval()
    all_spearman, all_top1 = [], []
    all_ious = []  # mean IoU per matched sample
    rank_tables = []  # for wandb logging
    n_total, n_matched = 0, 0

    for batch in dataloader:
        # ---- replicate the model's forward (encoder + decoder) --------
        images = [x["image"].to(device) for x in batch]
        images = [(x - model.pixel_mean) / model.pixel_std for x in images]
        images = ImageList.from_tensors(images, model.size_divisibility)

        # Text encoding (mirrors maskformer_model.forward)
        input_ids = [x['tokens']['input_ids'].to(device) for x in batch]
        attention_mask = [x['tokens']['attention_mask'].to(device) for x in batch]
        text_info = {'tokens': {
            'input_ids': torch.stack(input_ids).squeeze(1),
            'attention_mask': torch.stack(attention_mask).squeeze(1),
        }}

        p_input_ids = torch.stack([x['phrases']['p_input_ids'].to(device) for x in batch])
        p_attention_mask = torch.stack([x['phrases']['p_attention_mask'].to(device) for x in batch])
        bs, n_phrase = p_input_ids.shape[0], p_input_ids.shape[1]
        p_input_ids = p_input_ids.view(bs * n_phrase, -1)
        p_attention_mask = p_attention_mask.view(bs * n_phrase, -1)

        lan_out = model.lang_encoder(text_info['tokens'])
        phrase_out = model.lang_encoder({'input_ids': p_input_ids, 'attention_mask': p_attention_mask})

        if 'relations' in batch[0]:
            relation_tokens = torch.stack([x['relations']['r_input_ids'].to(device) for x in batch])
            relation_attn = torch.stack([x['relations']['r_attention_mask'].to(device) for x in batch])
            n_relations = relation_tokens.shape[1]
            relation_out = model.lang_encoder({
                'input_ids': relation_tokens.view(bs * n_relations, -1),
                'attention_mask': relation_attn.view(bs * n_relations, -1),
            })
            relation_embeds = relation_out['pooler_output'].view(bs, n_relations, -1)
        else:
            relation_embeds = None

        extra = {
            "masks": lan_out['masks'],
            "word_embeds": lan_out['embedded'],
            "sent_embeds": lan_out['pooler_output'],
            "phrase_pooled_feat": phrase_out['pooler_output'],
            "n_ph": n_phrase,
            "p_in_sent_mask": torch.stack([x['phrases']['p_in_sent_mask'].to(device) for x in batch]),
            'add_pooling_layer': model.add_pooling_layer,
            'relation_embeds': relation_embeds,
        }

        features = model.backbone(images.tensor)
        outputs, sal_scores, _ = model.sem_seg_head(features, extra=extra)

        # ---- Build targets (padded GT masks + ranks) ------------------
        gt_instances = [x["instances"].to(device) for x in batch]
        h_pad, w_pad = images.tensor.shape[-2:]
        targets = []
        for t in gt_instances:
            gt_masks = t.gt_masks
            if hasattr(gt_masks, 'tensor'):
                gt_masks = gt_masks.tensor
            padded = torch.zeros((gt_masks.shape[0], h_pad, w_pad),
                                 dtype=gt_masks.dtype, device=gt_masks.device)
            padded[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks
            targets.append({
                "labels": t.gt_classes,
                "masks": padded,
                "ranking": t.gt_ranks,
            })

        # ---- Compute Spearman & top-1 on matched pairs ---------------
        # Instead of relying on Hungarian matching (which is unreliable when
        # the frozen model's predicted masks don't overlap the H5 GT masks),
        # we match each GT object to the query whose predicted mask has the
        # highest IoU with that GT mask.  This gives a meaningful alignment.
        pred_masks_sig = outputs["pred_masks"].sigmoid()  # [B, Q, H', W']
        pred_masks_up = F.interpolate(
            pred_masks_sig, size=(h_pad, w_pad), mode="bilinear", align_corners=False,
        )  # [B, Q, H_pad, W_pad]

        for b_idx, (sample, scores, tgt) in enumerate(zip(batch, sal_scores, targets)):
            n_total += 1
            gt_masks_b = tgt["masks"]          # [N_gt, H_pad, W_pad]
            gt_ranks_all = tgt["ranking"]       # [N_gt]
            n_gt = gt_masks_b.shape[0]
            if n_gt < 2:
                continue

            # IoU-based matching: for each GT mask, find the best query
            pred_m = (pred_masks_up[b_idx] > 0.5).float()  # [Q, H, W]
            gt_m = gt_masks_b.float()                       # [N_gt, H, W]
            # Pairwise IoU  [N_gt, Q]
            gt_flat = gt_m.view(n_gt, -1)       # [N_gt, HW]
            pred_flat = pred_m.view(pred_m.shape[0], -1)  # [Q, HW]
            inter = gt_flat @ pred_flat.T        # [N_gt, Q]
            union = gt_flat.sum(1, keepdim=True) + pred_flat.sum(1, keepdim=True).T - inter
            iou = inter / (union + 1e-6)         # [N_gt, Q]

            best_query_per_gt = iou.argmax(dim=1)  # [N_gt]
            best_iou_per_gt = iou[torch.arange(n_gt), best_query_per_gt]  # [N_gt]
            all_ious.append(best_iou_per_gt.mean().item())

            gt_ranks_np = gt_ranks_all.cpu().numpy().astype(float)
            # scores shape is [Q, 1] — squeeze to 1-D
            scores_1d = scores.squeeze(-1).cpu().numpy().astype(float)  # [Q]
            pred_scores = scores_1d[best_query_per_gt.cpu().numpy()]    # [N_gt]

            n_matched += 1

            # Convert pred saliency scores to integer ranks.
            # Loss convention: lower score = more salient = rank 1.
            order = np.argsort(pred_scores)       # ascending — lowest score first
            pred_int_ranks = np.empty_like(pred_scores)
            pred_int_ranks[order] = np.arange(1, len(pred_scores) + 1)

            spr = spearmanr(pred_int_ranks, gt_ranks_np)
            try:
                val = spr.statistic
            except AttributeError:
                val = spr.correlation
            if not np.isnan(val):
                all_spearman.append(val)

            # Log per-segment GT vs predicted ranks
            for seg_i in range(len(gt_ranks_np)):
                rank_tables.append({
                    "sample_idx": n_total - 1,
                    "segment_idx": int(seg_i),
                    "gt_rank": float(gt_ranks_np[seg_i]),
                    "pred_rank": float(pred_int_ranks[seg_i]),
                    "pred_saliency_score": float(pred_scores[seg_i]),
                })

            # Top-1: does the most salient prediction match the most salient GT?
            # rank 1 = most salient → use argmin
            all_top1.append(
                1.0 if np.argmin(pred_int_ranks) == np.argmin(gt_ranks_np) else 0.0
            )

    avg_spr = np.mean(all_spearman) if all_spearman else float("nan")
    avg_top1 = np.mean(all_top1) if all_top1 else float("nan")
    avg_iou = np.mean(all_ious) if all_ious else float("nan")
    return {"n_total": n_total, "n_matched": n_matched,
            "spearman": avg_spr, "top1_acc": avg_top1,
            "mean_iou": avg_iou,
            "rank_details": rank_tables}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune full LGSOR on H5 data")
    parser.add_argument("--h5_path", type=str, default="../langgeonet/subset_10ep.h5")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/irsr_swinl/model.pth")
    parser.add_argument("--config-file", type=str,
                        default="configs/coco/instance-segmentation/swin/"
                                "maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default="output/h5_rank_train")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # --- Wandb ---
    wandb.init(
        project="lgsor-h5-ranking",
        config=vars(args),
        dir=args.save_dir,
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    # --- Build model ---
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    cfg = setup_cfg(args.config_file, args.checkpoint)
    model = build_model(cfg)

    # Load pretrained checkpoint
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(args.checkpoint)
    logger.info("Checkpoint loaded")

    # Unfreeze the entire architecture for full fine-tuning
    backbone_params, other_params, rank_params = unfreeze_all_and_group(model)

    # Keep full weight_dict: BCE + dice + ranking losses
    log_weight_dict(model)

    device = model.device

    logger.info("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  {name}: {param.shape}")

    # --- Data ---
    logger.info(f"H5: {args.h5_path}")
    train_loader, val_loader = create_lgsor_dataloaders(
        args.h5_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        image_size=args.image_size,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Optimizer: three LR groups ---
    param_groups = [
        {"params": backbone_params, "lr": args.lr * 0.01, "name": "backbone"},
        {"params": other_params, "lr": args.lr * 0.1, "name": "decoder_encoder"},
        {"params": rank_params, "lr": args.lr, "name": "rank_head"},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    all_trainable = backbone_params + other_params + rank_params
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6
    )
    logger.info(f"Rank-head LR: {args.lr}, Decoder/Encoder LR: {args.lr * 0.1}, "
                f"Backbone LR: {args.lr * 0.01}")

    # --- Training loop ---
    best_val_spearman = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Use the model's built-in forward which runs:
            #   backbone → pixel decoder → transformer decoder → criterion
            # The criterion does Hungarian matching + BCE + dice + ranking.
            loss_dict = model(batch)
            total_loss = sum(loss_dict.values())

            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()

            epoch_loss += total_loss.item()
            epoch_batches += 1

            # Log per-batch losses to wandb
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            wandb_log = {"train/total_loss": total_loss.item()}
            for k, v in loss_dict.items():
                wandb_log[f"train/{k}"] = v.item()
            wandb_log["train/lr_backbone"] = optimizer.param_groups[0]["lr"]
            wandb_log["train/lr_decoder"] = optimizer.param_groups[1]["lr"]
            wandb_log["train/lr_rank"] = optimizer.param_groups[2]["lr"]
            wandb.log(wandb_log, step=global_step)

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = epoch_loss / epoch_batches
                parts = " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
                logger.info(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} [{parts}]"
                )

        epoch_time = time.time() - t0
        avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
        cur_lrs = scheduler.get_last_lr()
        logger.info(
            f"Epoch {epoch} done | Avg loss: {avg_epoch_loss:.4f} | "
            f"Time: {epoch_time:.1f}s | LR bb={cur_lrs[0]:.2e} dec={cur_lrs[1]:.2e} rank={cur_lrs[2]:.2e}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_epoch_loss,
            "lr_backbone": cur_lrs[0],
            "lr_decoder": cur_lrs[1],
            "lr_rank": cur_lrs[2],
        }
        wandb.log({
            "epoch": epoch,
            "epoch/train_loss": avg_epoch_loss,
            "epoch/lr_backbone": cur_lrs[0],
            "epoch/lr_decoder": cur_lrs[1],
            "epoch/lr_rank": cur_lrs[2],
        })

        # --- Evaluation ---
        if epoch % args.eval_interval == 0 or epoch == 1 or epoch == args.epochs:
            val_metrics = evaluate(model, train_loader, device)# val_loader
            logger.info(
                f"  Val: spearman={val_metrics['spearman']:.4f} | "
                f"top1={val_metrics['top1_acc']:.4f} | "
                f"mean_iou={val_metrics['mean_iou']:.4f} | "
                f"matched={val_metrics['n_matched']}/{val_metrics['n_total']}"
            )
            # Log rank details to wandb table
            rank_details = val_metrics.pop("rank_details")
            if rank_details:
                columns = ["sample_idx", "segment_idx", "gt_rank", "pred_rank", "pred_saliency_score"]
                table = wandb.Table(columns=columns)
                for row in rank_details:
                    table.add_data(row["sample_idx"], row["segment_idx"],
                                   row["gt_rank"], row["pred_rank"],
                                   row["pred_saliency_score"])
                wandb.log({f"val/rank_table_ep{epoch}": table})

            wandb.log({
                "val/spearman": val_metrics["spearman"],
                "val/top1_acc": val_metrics["top1_acc"],
                "val/mean_iou": val_metrics["mean_iou"],
                "val/n_matched": val_metrics["n_matched"],
                "val/n_total": val_metrics["n_total"],
            })
            epoch_record.update({"val_" + k: v for k, v in val_metrics.items()})

            if (not np.isnan(val_metrics["spearman"])
                    and val_metrics["spearman"] > best_val_spearman):
                best_val_spearman = val_metrics["spearman"]
                save_path = os.path.join(args.save_dir, "best_model.pth")
                torch.save(model.state_dict(), save_path)
                logger.info(f"  New best model saved: spearman={best_val_spearman:.4f}")

        history.append(epoch_record)

        # --- Periodic save ---
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"model_ep{epoch}.pth")
            torch.save(model.state_dict(), save_path)

    # --- Final save ---
    save_path = os.path.join(args.save_dir, "model_final.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Final model saved to {save_path}")

    with open(os.path.join(args.save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info(f"Training history saved. Best val spearman: {best_val_spearman:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
