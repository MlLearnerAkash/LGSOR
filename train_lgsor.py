"""
Training script using the actual LGSOR model architecture.

Uses the LGSOR repo code as boilerplate:
  - ResNet-50 backbone (from detectron2)
  - MSDeformAttn pixel decoder
  - TransFusion (text-guided visual modulation)
  - Transformer decoder with QueryEncoder/TAVR
  - Multi-instance GAT graph → saliency rank scores
  - SetCriterion with pairwise rank loss

Key adaptation: masks come from the H5 dataset (pre-computed), so we
skip the Hungarian matching for mask assignment and directly use the
given masks + their intra-frame geodesic cost rankings.

Smoke test: train WITH vs WITHOUT language, compare ranking accuracy.

Usage:
  python train_lgsor.py --h5_path ../langgeonet/subset_10ep.h5 --smoke_test --epochs 20
"""

import os
import sys
import time
import argparse
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr

# Add LGSOR to path
LGSOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'LGSOR')
sys.path.insert(0, LGSOR_DIR)

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_backbone, build_sem_seg_head
from detectron2.structures import ImageList, BitMasks
from mask2former import add_maskformer2_config
from mask2former.modeling.language_encoder.bert import BertEncoder
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher

from costmap_predictor.LGSOR.h5_lgsor_dataset import create_lgsor_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build LGSOR model from detectron2 config (same as main.py)
# ---------------------------------------------------------------------------

def build_lgsor_cfg(image_size=1024, num_queries=100):
    """Build a detectron2 config matching LGSOR's maskformer2_R50_bs16_50ep.yaml"""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    # Load the R50 config from the LGSOR repo
    config_path = os.path.join(LGSOR_DIR, 'configs', 'coco', 'instance-segmentation',
                               'maskformer2_R50_bs16_50ep.yaml')
    if os.path.exists(config_path):
        cfg.merge_from_file(config_path)
    else:
        # Manually set essential params if config file not found
        cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
        cfg.MODEL.RESNETS.DEPTH = 50
        cfg.MODEL.RESNETS.STEM_TYPE = "basic"
        cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
        cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.BACKBONE.FREEZE_AT = 0

    # Override key params
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    cfg.MODEL.META_ARCHITECTURE = "MaskFormer"
    cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 6

    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = num_queries
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 10
    cfg.MODEL.MASK_FORMER.PRE_NORM = False
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 12544
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False

    cfg.MODEL.RELATION_HEAD.UNIT_NUMS = 8
    cfg.MODEL.RELATION_HEAD.LAYER_NUM = 2

    cfg.INPUT.IMAGE_SIZE = image_size

    # We don't train from pretrained weights
    cfg.MODEL.WEIGHTS = ""

    cfg.freeze()
    return cfg


class LGSORRankModel(nn.Module):
    """
    LGSOR model adapted for ranking with pre-given masks.
    
    Architecture (same as LGSOR paper):
      1. ResNet-50 backbone → multi-scale visual features
      2. MSDeformAttn pixel decoder → enhanced features + mask_features
      3. BERT language encoder (frozen) → word/sentence/phrase/relation embeddings
      4. TransFusion (text-guided visual modulation)
      5. Transformer decoder with QueryEncoder/TAVR
      6. Multi-instance GAT graph reasoning
      7. Saliency rank score prediction
      
    Masks come from the dataset, so we skip mask prediction loss
    and use given masks for the ranking supervision.
    """

    def __init__(self, cfg, use_language=True):
        super().__init__()
        self.cfg = cfg
        self.use_language = use_language
        self.device_param = nn.Parameter(torch.zeros(1))  # dummy for device detection

        # 1. Visual backbone (ResNet-50)
        self.backbone = build_backbone(cfg)

        # 2. Semantic segmentation head (pixel decoder + transformer decoder)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())

        # 3. Language encoder (BERT, frozen)
        self.lang_encoder = BertEncoder(
            'bert-base-uncased',
            use_checkpoint=True,
            add_pooling_layer=True
        )
        for param in self.lang_encoder.parameters():
            param.requires_grad = False

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Image normalization
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Forward pass following LGSOR's MaskFormer.forward().
        
        Args:
            batched_inputs: list of dicts with:
                - image: (C, H, W) tensor
                - tokens: {input_ids, attention_mask}
                - phrases: {p_input_ids, p_attention_mask, p_in_sent_mask}
                - relations: {r_input_ids, r_attention_mask}
                - instances: Instances with gt_masks, gt_classes, gt_ranks
        
        Returns:
            If training: dict of losses
            If eval: list of saliency scores per sample
        """
        # --- Image processing (same as MaskFormer) ---
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # --- Language processing (same as MaskFormer) ---
        bs = len(batched_inputs)

        input_ids = [x['tokens']['input_ids'].to(self.device) for x in batched_inputs]
        attention_mask = [x['tokens']['attention_mask'].to(self.device) for x in batched_inputs]
        text_info = {
            'tokens': {
                'input_ids': torch.stack(input_ids).squeeze(1),
                'attention_mask': torch.stack(attention_mask).squeeze(1),
            }
        }

        p_input_ids = torch.stack([x['phrases']['p_input_ids'].to(self.device) for x in batched_inputs])
        p_attention_mask = torch.stack([x['phrases']['p_attention_mask'].to(self.device) for x in batched_inputs])
        n_phrase = p_input_ids.shape[1]
        p_input_ids_flat = p_input_ids.view(bs * n_phrase, -1)
        p_attention_mask_flat = p_attention_mask.view(bs * n_phrase, -1)

        if self.use_language:
            with torch.no_grad():
                lan_out = self.lang_encoder(text_info['tokens'])
                phrase_out = self.lang_encoder({
                    'input_ids': p_input_ids_flat,
                    'attention_mask': p_attention_mask_flat
                })

            # Relation embeddings
            if 'relations' in batched_inputs[0] and batched_inputs[0]['relations'] is not None:
                relation_tokens = torch.stack([x['relations']['r_input_ids'].to(self.device) for x in batched_inputs])
                relation_attention_mask = torch.stack([x['relations']['r_attention_mask'].to(self.device) for x in batched_inputs])
                n_relations = relation_tokens.shape[1]
                with torch.no_grad():
                    relation_out = self.lang_encoder({
                        'input_ids': relation_tokens.view(bs * n_relations, -1),
                        'attention_mask': relation_attention_mask.view(bs * n_relations, -1)
                    })
                relation_embeds = relation_out['pooler_output'].view(bs, n_relations, -1)
            else:
                relation_embeds = None

            text_attention_mask = lan_out['masks']
            extra = {
                "masks": text_attention_mask,
                "word_embeds": lan_out['embedded'],
                "sent_embeds": lan_out['pooler_output'],
                "phrase_pooled_feat": phrase_out['pooler_output'],
                "n_ph": n_phrase,
                "p_in_sent_mask": torch.stack([x['phrases']['p_in_sent_mask'].to(self.device) for x in batched_inputs]),
                'add_pooling_layer': True,
                'relation_embeds': relation_embeds,
            }
        else:
            # No language: create zero embeddings
            hidden_dim = 768  # BERT hidden dim
            max_tokens = input_ids[0].shape[-1]
            extra = {
                "masks": torch.ones(bs, max_tokens, device=self.device),
                "word_embeds": torch.zeros(bs, max_tokens, hidden_dim, device=self.device),
                "sent_embeds": torch.zeros(bs, hidden_dim, device=self.device),
                "phrase_pooled_feat": torch.zeros(bs * n_phrase, hidden_dim, device=self.device),
                "n_ph": n_phrase,
                "p_in_sent_mask": torch.ones(bs, n_phrase, max_tokens, device=self.device),
                'add_pooling_layer': True,
                'relation_embeds': torch.zeros(bs, 18, hidden_dim, device=self.device),
            }

        # --- Visual backbone ---
        features = self.backbone(images.tensor)

        # --- Pixel decoder + Transformer decoder + TransFusion + TAVR + GAT ---
        # This calls sem_seg_head.forward() which internally does:
        #   pixel_decoder → mask_features + multi_scale_features
        #   transformer_decoder → outputs, sal_scores, side_rank_scores
        outputs, sal_scores, side_rank_score = self.sem_seg_head(features, extra=extra)

        if self.training:
            # Prepare targets from given masks (same format as MaskFormer.prepare_targets)
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            h_pad, w_pad = images.tensor.shape[-2:]
            targets = []
            for t in gt_instances:
                gt_masks = t.gt_masks
                if isinstance(gt_masks, BitMasks):
                    gt_masks = gt_masks.tensor
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad),
                                           dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks
                targets.append({
                    "labels": t.gt_classes,
                    "masks": padded_masks,
                    "ranking": t.gt_ranks,
                })

            # Compute rank loss only (skip mask/class loss for given masks)
            rank_loss = self._compute_rank_loss(sal_scores, targets, outputs)

            # Also compute side rank loss if available
            side_loss = torch.tensor(0.0, device=self.device)
            if side_rank_score is not None:
                for i, srs in enumerate(side_rank_score):
                    side_loss = side_loss + self._compute_rank_loss(srs, targets, outputs)
                side_loss = side_loss / len(side_rank_score)

            losses = {
                'loss_ranks_final': rank_loss * 5.0,  # weight from LGSOR config
            }
            if side_rank_score is not None:
                losses['loss_ranks_side'] = side_loss * 3.0

            return losses, sal_scores
        else:
            return sal_scores

    def _compute_rank_loss(self, sal_scores, targets, outputs):
        """Compute pairwise rank loss using LGSOR's rankloss_compute.
        
        Since masks are given (not predicted), we use identity matching:
        each GT object i maps to query i (using Hungarian matching on the given masks).
        """
        # Use Hungarian matching to align queries to GT objects
        # (necessary because the transformer decoder outputs 100 queries but
        # we may have fewer GT objects)
        matcher = HungarianMatcher(
            cost_class=2.0, cost_mask=5.0, cost_dice=5.0,
            num_points=self.cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        outputs_for_match = {
            'pred_logits': outputs['pred_logits'],
            'pred_masks': outputs['pred_masks'],
        }
        indices = matcher(outputs_for_match, targets)

        # Compute rank loss for each image using matched indices
        rank_losses = []
        for t, scores, (src_idx, tgt_idx) in zip(targets, sal_scores, indices):
            gt_ranks = t["ranking"][tgt_idx]
            pred_scores = scores[src_idx]
            rank_loss = self._rankloss_compute(gt_ranks, pred_scores)
            rank_losses.append(rank_loss)

        total_loss = sum(rank_losses) / max(len(rank_losses), 1)
        return total_loss

    @staticmethod
    def _rankloss_compute(rank_labels, saliency_score):
        """Exact pairwise rank loss from LGSOR's SetCriterion.rankloss_compute."""
        N = len(rank_labels)
        if N < 2:
            return torch.tensor(0.0, device=saliency_score.device)
        saliency_score = saliency_score.reshape(-1)
        S1, S2 = torch.meshgrid(saliency_score, saliency_score, indexing='ij')
        S = -S1 + S2
        R1, R2 = torch.meshgrid(rank_labels.float(), rank_labels.float(), indexing='ij')
        R = (R1 - R2)
        R_sign = R.clone()
        R_sign[R > 0] = 1
        R_sign[R < 0] = -1
        S = S * R_sign
        S = torch.log(1 + torch.exp(S))
        S[R_sign == 0] = 0
        S = torch.triu(S, 1)
        B = torch.abs(R.float())
        Wr_m = torch.sum(torch.arange(1, N, device=saliency_score.device).float() *
                         torch.arange(N - 1, 0, -1, device=saliency_score.device).float())
        B = B / Wr_m
        S = S * B
        relation_loss = torch.sum(S)
        return relation_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ranking_metrics(all_pred_scores, all_gt_ranks):
    pairwise_accs = []
    spearman_corrs = []

    for pred, gt in zip(all_pred_scores, all_gt_ranks):
        if len(pred) < 2:
            continue
        correct = total = 0
        for i in range(len(pred)):
            for j in range(i + 1, len(pred)):
                if gt[i] == gt[j]:
                    continue
                total += 1
                pred_order = pred[i] > pred[j]
                gt_order = gt[i] < gt[j]
                correct += int(pred_order == gt_order)
        if total > 0:
            pairwise_accs.append(correct / total)
        if len(pred) >= 3:
            c, _ = spearmanr(-pred, gt)
            if not np.isnan(c):
                spearman_corrs.append(c)

    return {
        'pairwise_acc': np.mean(pairwise_accs) if pairwise_accs else 0.0,
        'spearman': np.mean(spearman_corrs) if spearman_corrs else 0.0,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()

        loss_dict, sal_scores = model(batch)
        loss = sum(loss_dict.values())

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Skipping batch {batch_idx} with nan/inf loss")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)  # LGSOR uses 0.01
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 5 == 0:
            loss_str = " ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
            logger.info(f"Epoch {epoch} [{batch_idx}/{len(loader)}] {loss_str}")

    return {'loss': total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_pred_scores = []
    all_gt_ranks = []

    for batch in loader:
        # Run in training mode briefly to get loss
        model.train()
        loss_dict, sal_scores = model(batch)
        model.eval()

        loss = sum(loss_dict.values())
        total_loss += loss.item()
        n_batches += 1

        # Extract matched predictions for metrics
        # sal_scores is list of [num_queries, 1] tensors
        # Use the top-K scores matching GT objects
        for i, sample in enumerate(batch):
            gt_ranks = sample['integer_ranks'].numpy()
            n_obj = len(gt_ranks)
            # Get top n_obj predictions by score magnitude
            scores = sal_scores[i].squeeze(-1).cpu().numpy()
            # Use the first n_obj scores (after matching they should align)
            top_idx = np.argsort(-scores)[:n_obj]
            pred = scores[top_idx]
            all_pred_scores.append(pred)
            all_gt_ranks.append(gt_ranks)

    metrics = compute_ranking_metrics(all_pred_scores, all_gt_ranks)
    metrics['loss'] = total_loss / max(n_batches, 1)
    return metrics


def train(args, use_language=True, tag=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}, use_language={use_language}")

    # Data
    train_loader, val_loader = create_lgsor_dataloaders(
        h5_path=args.h5_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        image_size=args.image_size,
    )

    # Build LGSOR config
    cfg = build_lgsor_cfg(image_size=args.image_size, num_queries=args.num_queries)

    # Build model
    model = LGSORRankModel(cfg, use_language=use_language).to(device)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model [{tag}]: {total_params:,} total, {trainable_params:,} trainable "
                f"({100*trainable_params/total_params:.1f}%)")

    # Optimizer (like LGSOR config)
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = optim.AdamW([
        {'params': head_params, 'lr': args.lr},
        {'params': backbone_params, 'lr': args.lr * 0.1},
    ], weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_acc = 0.0
    history = defaultdict(list)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, device)

        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{args.epochs} [{tag}] ({elapsed:.1f}s) — "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"pairwise_acc={val_metrics['pairwise_acc']:.4f} "
            f"spearman={val_metrics['spearman']:.4f}"
        )

        for k, v in train_metrics.items():
            history[f'train_{k}'].append(v)
        for k, v in val_metrics.items():
            history[f'val_{k}'].append(v)

        if val_metrics['pairwise_acc'] > best_val_acc:
            best_val_acc = val_metrics['pairwise_acc']
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, f"best_model_{tag}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_acc': best_val_acc,
                    'use_language': use_language,
                }, ckpt_path)
                logger.info(f"Saved best model to {ckpt_path}")

    return history, best_val_acc


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(args):
    logger.info("=" * 60)
    logger.info("SMOKE TEST: Language vs No-Language (LGSOR architecture)")
    logger.info("=" * 60)

    logger.info("\n>>> Training WITH language conditioning <<<")
    hist_lang, best_lang = train(args, use_language=True, tag="with_lang")

    logger.info("\n>>> Training WITHOUT language conditioning <<<")
    hist_nolang, best_nolang = train(args, use_language=False, tag="no_lang")

    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"WITH language:    best pairwise_acc = {best_lang:.4f}")
    logger.info(f"WITHOUT language: best pairwise_acc = {best_nolang:.4f}")
    logger.info(f"Improvement: {best_lang - best_nolang:+.4f} "
                f"({100*(best_lang - best_nolang)/max(best_nolang, 1e-6):+.1f}%)")

    if best_lang > best_nolang:
        logger.info("Language conditioning HELPS ranking accuracy")
    elif best_lang < best_nolang:
        logger.info("Language conditioning did NOT help (need more data/epochs)")
    else:
        logger.info("No difference detected")

    logger.info("\nPer-epoch pairwise accuracy comparison:")
    logger.info(f"{'Epoch':>5} | {'With Lang':>10} | {'No Lang':>10} | {'Delta':>10}")
    logger.info("-" * 45)
    n_epochs = min(len(hist_lang.get('val_pairwise_acc', [])),
                   len(hist_nolang.get('val_pairwise_acc', [])))
    for i in range(n_epochs):
        wl = hist_lang['val_pairwise_acc'][i]
        nl = hist_nolang['val_pairwise_acc'][i]
        logger.info(f"{i+1:>5} | {wl:>10.4f} | {nl:>10.4f} | {wl-nl:>+10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LGSOR Model Training for Object Ranking"
    )
    parser.add_argument('--h5_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2.5e-5)  # LGSOR default
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='checkpoints_lgsor')
    parser.add_argument('--smoke_test', action='store_true')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.smoke_test:
        run_smoke_test(args)
    else:
        train(args, use_language=True, tag="full")


if __name__ == '__main__':
    main()
