"""
Dataset adapter: converts H5 episodic data to the dict format expected by
the LGSOR MaskFormer model.

Each sample is a dict with keys:
  - image: (C, H, W) tensor
  - tokens: {input_ids, attention_mask}
  - phrases: {p_input_ids, p_attention_mask, p_in_sent_mask}
  - relations: {r_input_ids, r_attention_mask}
  - instances: detectron2.structures.Instances with gt_masks, gt_classes, gt_ranks
  - height, width, file_name
"""

import os
import sys
import random
import pickle
import types

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import BertTokenizer
from detectron2.structures import Instances, BitMasks

# ---------------------------------------------------------------------------
# Magnum mock
# ---------------------------------------------------------------------------

def _install_magnum_mock():
    if '_magnum' in sys.modules:
        return
    magnum_mock = types.ModuleType('_magnum')

    class _GenericMock:
        def __init__(self, *a, **kw):
            self.data = list(a)
        def __setstate__(self, s):
            self.data = s
        def __getstate__(self):
            return self.data

    for name in [
        'Vector2', 'Vector2i', 'Vector3', 'Vector3i', 'Vector4',
        'Quaternion', 'Matrix3x3', 'Matrix4',
        'Color3', 'Color4', 'Range1D', 'Range2D', 'Range3D',
    ]:
        setattr(magnum_mock, name, type(name, (_GenericMock,), {}))
    sys.modules['_magnum'] = magnum_mock


_install_magnum_mock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_rle_mask(rle_dict, H, W):
    try:
        if not isinstance(rle_dict, dict) or 'counts' not in rle_dict:
            return np.zeros((H, W), dtype=bool)
        counts = rle_dict['counts']
        if isinstance(counts, str):
            counts = counts.encode('utf-8')
            rle_dict = {'size': rle_dict['size'], 'counts': counts}
        comp = mask_utils.frPyObjects([rle_dict], H, W)
        dec = mask_utils.decode(comp)
        m = (dec[..., 0] if dec.ndim == 3 else dec).astype(bool)
        return m
    except Exception:
        return np.zeros((H, W), dtype=bool)


def extract_nouns_simple(instruction):
    """Simple noun extraction from instruction for entity cues.
    Returns list of (phrase, start_pos, end_pos) tuples."""
    # Simple heuristic: use individual words as entity cues
    words = instruction.lower().split()
    # Filter common stop words
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'shall', 'can',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'not',
        'so', 'very', 'just', 'about', 'up', 'down', 'here', 'there',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'only',
        'own', 'same', 'than', 'too', 'it', 'its', 'this', 'that',
        'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our',
        'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
        'them', 'their', 'what', 'which', 'who', 'whom',
        'go', 'turn', 'walk', 'left', 'right', 'straight', 'stop',
        'continue', 'past', 'until', 'take',
    }
    nouns = []
    for w in words:
        w_clean = w.strip('.,!?;:')
        if w_clean and w_clean not in stop_words and len(w_clean) > 2:
            nouns.append(w_clean)
    return nouns[:25]  # cap at max phrases


def extract_relations_simple(instruction):
    """Simple relation extraction from instruction for relation cues."""
    relation_keywords = [
        'next to', 'near', 'beside', 'behind', 'in front of',
        'above', 'below', 'between', 'around', 'toward',
        'away from', 'across', 'along', 'through', 'inside',
        'outside', 'facing', 'past', 'before', 'after',
    ]
    text = instruction.lower()
    found = []
    for rel in relation_keywords:
        if rel in text:
            found.append(rel)
    # Also add directional words
    for w in text.split():
        w_clean = w.strip('.,!?;:')
        if w_clean in {'left', 'right', 'up', 'down', 'forward', 'back', 'ahead'}:
            found.append(w_clean)
    if not found:
        found = ['near']  # default
    return found[:18]  # cap at max relations


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class H5LGSORDataset(Dataset):
    """
    Converts H5 episodic data into the dict format expected by LGSOR's MaskFormer.
    
    Key difference from original LGSOR: masks come from the dataset (pre-computed)
    instead of being predicted by the model.
    """

    def __init__(self, h5_path, episode_ids=None, image_size=1024,
                 max_objects=100, max_frames_per_episode=200,
                 max_tokens=256, num_phrases=25, num_relations=18,
                 phrase_seq_len=10):
        super().__init__()
        self.h5_path = h5_path
        self.image_size = image_size
        self.max_objects = max_objects
        self.max_tokens = max_tokens
        self.num_phrases = num_phrases
        self.num_relations = num_relations
        self.phrase_seq_len = phrase_seq_len
        self._h5 = None

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Detectron2 pixel mean/std (ImageNet)
        self.pixel_mean = torch.tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.120, 57.375]).view(3, 1, 1)

        # Pre-index samples
        self.samples = []
        self._episode_cache = {}

        with h5py.File(h5_path, 'r') as hf:
            all_keys = sorted(hf.keys())
            ep_keys = episode_ids if episode_ids is not None else all_keys

        for ep_id in ep_keys:
            ep_data = self._load_episode(ep_id)
            if ep_data is None:
                continue
            G = ep_data['graph']
            all_paths = G.graph.get('all_paths_lengths')
            if all_paths is None:
                continue

            all_nodes = list(G.nodes)
            n_frames = len(ep_data['frame_data'])
            if n_frames > max_frames_per_episode:
                continue

            for local_i, fd in enumerate(ep_data['frame_data']):
                frame_idx = int(fd['frame_idx'])
                frame_nodes = [n for n in all_nodes
                               if G.nodes[n]['map'][0] == frame_idx]
                if len(frame_nodes) < 2:
                    continue
                self.samples.append({
                    'episode_id': ep_id,
                    'local_frame_idx': local_i,
                    'frame_idx': frame_idx,
                    'frame_nodes': frame_nodes,
                })

        self._episode_cache = {}
        print(f"[H5LGSORDataset] {len(self.samples)} frames from "
              f"{len(set(s['episode_id'] for s in self.samples))} episodes")

    def _load_episode(self, ep_id):
        if ep_id in self._episode_cache:
            return self._episode_cache[ep_id]
        try:
            if self._h5 is None:
                self._h5 = h5py.File(self.h5_path, 'r')
            hf = self._h5
            if ep_id not in hf:
                return None
            instruction = hf[ep_id]['instruction'][()]
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')
            G = pickle.loads(hf[ep_id]['graph'][()])
            frame_data = []
            frames_grp = hf[ep_id]['frames']
            for fk in sorted(frames_grp.keys()):
                fd = {
                    'frame_idx': int(frames_grp[fk]['frame_idx'][()]),
                    'rgb': frames_grp[fk]['rgb'][()],
                }
                frame_data.append(fd)
            ep_data = {
                'instruction': instruction,
                'graph': G,
                'frame_data': frame_data,
            }
            self._episode_cache[ep_id] = ep_data
            return ep_data
        except Exception as e:
            print(f"Warning: failed to load episode {ep_id}: {e}")
            return None

    def _tokenize_text(self, instruction):
        """Tokenize instruction and extract phrase/relation cues like LGSOR."""
        # Full sentence tokens
        tokens = self.tokenizer(
            instruction,
            padding='max_length',
            truncation=True,
            max_length=self.max_tokens,
            return_tensors='pt',
        )

        # Entity cues (phrases = nouns)
        nouns = extract_nouns_simple(instruction)
        tokenized_phrases = []
        phrase_masks = []
        for noun in nouns:
            tok = self.tokenizer(
                noun,
                padding='max_length',
                max_length=self.phrase_seq_len,
                truncation=True,
                return_tensors='pt',
            )
            tokenized_phrases.append(tok['input_ids'][0])
            phrase_masks.append(tok['attention_mask'][0])

        # Pad to num_phrases
        for _ in range(len(nouns), self.num_phrases):
            tok = self.tokenizer(
                "",
                padding='max_length',
                max_length=self.phrase_seq_len,
                return_tensors='pt',
            )
            tokenized_phrases.append(tok['input_ids'][0])
            phrase_masks.append(tok['attention_mask'][0])

        # Build p_in_sent_mask (which tokens in the sentence correspond to each phrase)
        p_in_sent_mask = []
        for j in range(self.num_phrases):
            mask = torch.ones_like(tokens['attention_mask'][0])
            # Find phrase position in sentence tokens (approximate)
            if j < len(nouns):
                phrase_tokens = self.tokenizer.encode(nouns[j], add_special_tokens=False)
                sent_ids = tokens['input_ids'][0].tolist()
                # Search for phrase tokens in sentence
                for start in range(len(sent_ids) - len(phrase_tokens) + 1):
                    if sent_ids[start:start + len(phrase_tokens)] == phrase_tokens:
                        mask[start:start + len(phrase_tokens)] = 0
                        break
            p_in_sent_mask.append(mask)

        # Relation cues
        relations = extract_relations_simple(instruction)
        tokenized_relations = []
        relation_masks = []
        for rel in relations:
            tok = self.tokenizer(
                rel,
                padding='max_length',
                max_length=self.phrase_seq_len,
                truncation=True,
                return_tensors='pt',
            )
            tokenized_relations.append(tok['input_ids'][0])
            relation_masks.append(tok['attention_mask'][0])

        for _ in range(len(relations), self.num_relations):
            tok = self.tokenizer(
                "",
                padding='max_length',
                max_length=self.phrase_seq_len,
                return_tensors='pt',
            )
            tokenized_relations.append(tok['input_ids'][0])
            relation_masks.append(tok['attention_mask'][0])

        return {
            'tokens': {
                'input_ids': tokens['input_ids'],      # [1, max_tokens]
                'attention_mask': tokens['attention_mask'],
            },
            'phrases': {
                'p_input_ids': torch.stack(tokenized_phrases),       # [num_phrases, phrase_seq_len]
                'p_attention_mask': torch.stack(phrase_masks),
                'p_in_sent_mask': torch.stack(p_in_sent_mask),       # [num_phrases, max_tokens]
            },
            'relations': {
                'r_input_ids': torch.stack(tokenized_relations),     # [num_relations, phrase_seq_len]
                'r_attention_mask': torch.stack(relation_masks),
            },
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ep_data = self._load_episode(sample['episode_id'])
        G = ep_data['graph']
        fd = ep_data['frame_data'][sample['local_frame_idx']]
        frame_nodes = sample['frame_nodes']

        all_nodes = list(G.nodes)
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        all_paths = G.graph['all_paths_lengths']

        # Get intra-frame cost sub-matrix
        indices = [node_to_idx[n] for n in frame_nodes]
        intra_costs = all_paths[np.ix_(indices, indices)].astype(np.float32)

        # Per-object rank score
        K = len(frame_nodes)
        rank_scores = np.zeros(K, dtype=np.float32)
        for i in range(K):
            others = np.delete(intra_costs[i], i)
            rank_scores[i] = others.mean() if others.size > 0 else 0.0

        # Min-max normalise
        rs_min, rs_max = rank_scores.min(), rank_scores.max()
        if rs_max - rs_min > 1e-6:
            rank_scores = (rank_scores - rs_min) / (rs_max - rs_min)
        else:
            rank_scores = np.zeros_like(rank_scores)

        # Integer ranks (1-indexed, 1 = most salient = lowest cost)
        sorted_indices = np.argsort(rank_scores)
        integer_ranks = np.zeros(K, dtype=np.int64)
        for rank_pos, obj_idx in enumerate(sorted_indices):
            integer_ranks[obj_idx] = rank_pos + 1

        # Decode masks and image
        rgb = fd['rgb']
        H_orig, W_orig = rgb.shape[:2]
        masks = []
        for n in frame_nodes:
            rle = G.nodes[n].get('seg', G.nodes[n].get('segmentation'))
            if rle is not None:
                m = decode_rle_mask(rle, H_orig, W_orig)
            else:
                m = np.zeros((H_orig, W_orig), dtype=bool)
            masks.append(m)
        masks = np.stack(masks, axis=0)  # [K, H, W]

        # Truncate to max_objects
        if K > self.max_objects:
            keep_idx = np.argsort(-rank_scores)[:self.max_objects]
            keep_idx = np.sort(keep_idx)
            masks = masks[keep_idx]
            rank_scores = rank_scores[keep_idx]
            integer_ranks = integer_ranks[keep_idx]
            K = self.max_objects
            # Re-normalise
            rs_min, rs_max = rank_scores.min(), rank_scores.max()
            if rs_max - rs_min > 1e-6:
                rank_scores = (rank_scores - rs_min) / (rs_max - rs_min)
            sorted_indices = np.argsort(rank_scores)
            integer_ranks = np.zeros(K, dtype=np.int64)
            for rank_pos, obj_idx in enumerate(sorted_indices):
                integer_ranks[obj_idx] = rank_pos + 1

        # Resize image
        pil_img = Image.fromarray(rgb.astype(np.uint8))
        pil_img = pil_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(pil_img, dtype=np.float32)  # [H, W, 3]
        # Detectron2 uses BGR and float
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))  # [3, H, W] RGB float

        # Resize masks
        masks_t = torch.from_numpy(masks.astype(np.float32)).unsqueeze(1)
        masks_resized = torch.nn.functional.interpolate(
            masks_t, size=(self.image_size, self.image_size), mode='nearest'
        ).squeeze(1).bool()  # [K, H', W']

        # Build detectron2 Instances
        instances = Instances((self.image_size, self.image_size))
        instances.gt_masks = masks_resized
        instances.gt_classes = torch.zeros(K, dtype=torch.long)  # all same class (salient object)
        instances.gt_ranks = torch.from_numpy(integer_ranks)

        # Tokenize text  
        text_data = self._tokenize_text(ep_data['instruction'])

        result = {
            'image': img_tensor,
            'height': self.image_size,
            'width': self.image_size,
            'file_name': f"{sample['episode_id']}_frame{sample['frame_idx']}",
            'instances': instances,
            'annotations': [],  # not used directly
            'tokens': text_data['tokens'],
            'phrases': text_data['phrases'],
            'relations': text_data['relations'],
            # Extra for ranking evaluation
            'rank_scores': torch.from_numpy(rank_scores),
            'integer_ranks': torch.from_numpy(integer_ranks),
            'instruction': ep_data['instruction'],
        }
        return result


def lgsor_collate_fn(batch):
    """Identity collate — LGSOR expects list of dicts."""
    return batch


def create_lgsor_dataloaders(h5_path, batch_size=2, num_workers=0,
                             val_split=0.2, seed=42, image_size=1024):
    """Create train/val dataloaders splitting at episode level."""
    with h5py.File(h5_path, 'r') as hf:
        all_keys = sorted(hf.keys())

    rng = random.Random(seed)
    keys = list(all_keys)
    rng.shuffle(keys)

    split_idx = int(len(keys) * (1 - val_split))
    train_ids = keys[:split_idx]
    val_ids = keys[split_idx:]

    print(f"Train episodes: {len(train_ids)}, Val episodes: {len(val_ids)}")

    train_ds = H5LGSORDataset(h5_path, episode_ids=train_ids,
                               image_size=image_size)
    val_ds = H5LGSORDataset(h5_path, episode_ids=val_ids,
                             image_size=image_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=lgsor_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=lgsor_collate_fn,
        pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader
