"""
Copyright 2024 LY Corporation
LY Corporation licenses this file to you under the CC BY-NC 4.0
(the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:
    https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""

import logging
import os
import sys
from os.path import join as pjoin
from thop import profile

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from dataset import TextMotionPartDataset
from models.clip import ClipModel

log = logging.getLogger(__name__)

def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

@hydra.main(version_base=None, config_name="test_config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    saved_cfg = OmegaConf.load(pjoin(cfg.checkpoints_dir, ".hydra/config.yaml"))
    print(OmegaConf.to_yaml(saved_cfg))
    test_dataloader = prepare_test_dataset(saved_cfg)
    model, tokenizer = prepare_test_model(saved_cfg)
    part_enhanced = cfg.eval.part_enhanced
    eval_part(saved_cfg, test_dataloader, model, tokenizer, part_enhanced=part_enhanced)


def prepare_test_dataset(cfg):
    mean = np.load(pjoin(cfg.dataset.data_root, "Mean_raw.npy"))
    std = np.load(pjoin(cfg.dataset.data_root, "Std_raw.npy"))

    if cfg.eval.eval_train:
        test_split_file = pjoin(cfg.dataset.data_root, "train.txt")
    else:
        test_split_file = pjoin(cfg.dataset.data_root, "test.txt")
    test_dataset = TextMotionPartDataset(
        cfg,
        mean,
        std,
        test_split_file,
        eval_mode=True,
        patch_size=cfg.train.patch_size,
        fps=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=16
    )
    return test_dataloader

def prepare_test_model(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    motion_encoder_alias = cfg.model.motion_encoder
    text_encoder_alias = cfg.model.text_encoder
    motion_embedding_dims: int = 768
    text_embedding_dims: int = 768
    projection_dims: int = 256

    tokenizer = AutoTokenizer.from_pretrained(
        text_encoder_alias, TOKENIZERS_PARALLELISM=True
    )
    model = ClipModel(
        motion_encoder_alias=motion_encoder_alias,
        text_encoder_alias=text_encoder_alias,
        motion_embedding_dims=motion_embedding_dims,
        text_embedding_dims=text_embedding_dims,
        projection_dims=projection_dims,
        patch_size=cfg.train.patch_size,
        part_contrast=cfg.train.part_weight,
    )

    if cfg.eval.use_best_model:
        model_path = pjoin(cfg.checkpoints_dir, "best_model.pt")
    else:
        model_path = pjoin(cfg.checkpoints_dir, "last_model.pt")

    print(model_path)
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.to(device)

    return model, tokenizer

def eval_part(cfg, test_dataloader, model, tokenizer=None, verbose=True, part_enhanced=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_pair = dict()

    all_imgs_feat = []
    all_captions_feat = []

    all_img_idxs = []
    all_captions = []
    all_part_tfeat = []
    all_part_mfeat = []
    part_masks = []
    name_list = []
    all_len_list = []
    step = 0
    cnt_len = 0
    attn_list = []
    PART_NUM = 5
    with torch.no_grad():
        model.eval()
        test_pbar = tqdm(test_dataloader, leave=False)
        avg_bloss, avg_ploss = 0.0, 0.0
        for batch in test_pbar:
            step += 1
            texts, motions, part_text, part_mask, m_length, img_indexs = batch
            texts = list(texts)
            part_text = list(part_text)
            for i in range(PART_NUM):
                part_text[i] = list(part_text[i])

            motions = motions.to(device)
            part_text = np.array(part_text).transpose(1,0).flatten().tolist()

            texts_token = tokenizer(
                texts, padding=True, return_tensors="pt"
            ).to(device)
            part_texts_token = tokenizer(
                part_text, padding=True, return_tensors="pt"
            ).to(device)
            all_len_list.append(m_length)
            attn_save = False

            motion_features, text_features, part_text_features, bloss, ploss = model(motions, texts_token, part_texts_token, part_mask.cuda(),length=m_length.cuda())
            if attn_save:
                attn = model.encode_motion(motions, length=m_length.cuda(), return_attn=True)[1]
                attn_list.append(attn.cpu().numpy())
            part_motion_features = torch.stack(motion_features[1:], dim=1)
            motion_features = motion_features[0]

            avg_bloss += bloss.item()
            avg_ploss += ploss.item()
            part_masks.append(part_mask)
            for i in range(motion_features.size(0)):
                
                all_imgs_feat.append(motion_features[i].cpu().numpy())
                all_captions_feat.append(text_features[i].cpu().numpy())
                all_part_tfeat.append(part_text_features[i].cpu().numpy())
                all_part_mfeat.append(part_motion_features[i].cpu().numpy())
                all_captions.append(texts[i])
                all_img_idxs.append(img_indexs[i].item())
                name_list.append(test_dataloader.dataset.name_list[img_indexs[i].item()])
        avg_bloss /= len(test_dataloader)
        avg_ploss /= len(test_dataloader)

    all_captions = np.array(all_captions)

    for img_idx, caption in zip(all_img_idxs, all_captions):
        dataset_pair[img_idx] = np.where(all_captions == caption)[0]

    all_imgs_feat = np.vstack(all_imgs_feat)
    all_captions_feat = np.vstack(all_captions_feat)
    all_part_mfeat = np.stack(all_part_mfeat, axis=0)
    all_part_tfeat = np.stack(all_part_tfeat, axis=0)
    part_masks = torch.cat(part_masks, dim=0).cpu().numpy()
    if attn_save:
        attn_list = np.concatenate(attn_list, axis=0)
    # match test queries to target motions, get nearest neighbors
    sims_t2m = 100 * all_captions_feat.dot(all_imgs_feat.T)
    all_img_idxs = np.array(all_img_idxs)

    part_sim_scores = np.zeros_like(sims_t2m)
    if part_enhanced:
        for i in range(PART_NUM):
            part_mfeat = all_part_mfeat[:, i]
            part_tfeat = all_part_tfeat[:, i]
            part_sim = 100 * part_tfeat.dot(part_mfeat.T)
            part_sim[part_masks[:,i].astype(bool),:] = 0
            part_sim[:, part_masks[:,i].astype(bool)] = 0
            sims_t2m += 0.1 * part_sim
            part_sim_scores += 0.1 * part_sim
                    
    t2m_r1 = 0
    # Text->Motion
    ranks = np.zeros(sims_t2m.shape[0])
    for index, score in enumerate(tqdm(sims_t2m)):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in dataset_pair[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    for k in [1, 2, 3, 5, 10]:
        # Compute metrics
        r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
        if k == 1:
            t2m_r1 = r
        if verbose:
            log.info(f"t2m_recall_top{k}_correct_composition: {r:.2f}")
    if verbose:
        log.info(f"t2m_recall_median_correct_composition: {np.median(ranks)+1:.2f}")

    # match motions queries to target texts, get nearest neighbors
    sims_m2t = sims_t2m.T

    m2t_r1 = 0
    # Motion->Text
    ranks = np.zeros(sims_m2t.shape[0])
    for index, score in enumerate(tqdm(sims_m2t)):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in dataset_pair[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    for k in [1, 2, 3, 5, 10]:
        # Compute metrics
        r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
        if k == 1:
            m2t_r1 = r
        if verbose:
            log.info(f"m2t_recall_top{k}_correct_composition: {r:.2f}")
    if verbose:
        log.info(f"m2t_recall_median_correct_composition: {np.median(ranks)+1:.2f}")

    return t2m_r1, m2t_r1, avg_bloss, avg_ploss

def eval(cfg, test_dataloader, model, tokenizer=None, verbose=True, truncation=False,max_length=80):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_pair = dict()

    all_imgs_feat = []
    all_captions_feat = []

    all_img_idxs = []
    all_captions = []

    step = 0
    with torch.no_grad():
        model.eval()
        test_pbar = tqdm(test_dataloader, leave=False)
        for batch in test_pbar:
            step += 1
            texts, motions, _, _, m_length, img_indexs = batch
            motions = motions.to(device)

            texts_token = tokenizer(
                texts, padding=True, return_tensors="pt",truncation=truncation,max_length=max_length
            ).to(device)

            motion_features = model.encode_motion(motions)[0]
            text_features = model.encode_text(texts_token)

            # normalized features
            motion_features = motion_features / motion_features.norm(
                dim=1, keepdim=True
            )
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            for i in range(motion_features.size(0)):
                all_imgs_feat.append(motion_features[i].cpu().numpy())
                all_captions_feat.append(text_features[i].cpu().numpy())

                all_captions.append(texts[i])
                all_img_idxs.append(img_indexs[i])

    all_captions = np.array(all_captions)
    for img_idx, caption in zip(all_img_idxs, all_captions):
        dataset_pair[int(img_idx)] = np.where(all_captions == caption)[0]
    

    all_imgs_feat = np.vstack(all_imgs_feat)
    all_captions_feat = np.vstack(all_captions_feat)

    # match test queries to target motions, get nearest neighbors
    sims_t2m = 100 * all_captions_feat.dot(all_imgs_feat.T)

    t2m_r1 = 0
    # Text->Motion
    ranks = np.zeros(sims_t2m.shape[0])
    for index, score in enumerate(tqdm(sims_t2m)):
        index = int(index)
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in dataset_pair[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    for k in [1, 2, 3, 5, 10]:
        # Compute metrics
        r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
        if k == 1:
            t2m_r1 = r
        if verbose:
            log.info(f"t2m_recall_top{k}_correct_composition: {r:.2f}")
    if verbose:
        log.info(f"t2m_recall_median_correct_composition: {np.median(ranks)+1:.2f}")

    # match motions queries to target texts, get nearest neighbors
    sims_m2t = sims_t2m.T

    m2t_r1 = 0
    # Motion->Text
    ranks = np.zeros(sims_m2t.shape[0])
    for index, score in enumerate(tqdm(sims_m2t)):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in dataset_pair[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    for k in [1, 2, 3, 5, 10]:
        # Compute metrics
        r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
        if k == 1:
            m2t_r1 = r
        if verbose:
            log.info(f"m2t_recall_top{k}_correct_composition: {r:.2f}")
    if verbose:
        log.info(f"m2t_recall_median_correct_composition: {np.median(ranks)+1:.2f}")

    return t2m_r1, m2t_r1, torch.tensor(0.0), torch.tensor(0.0)


if __name__ == "__main__":
    main()
