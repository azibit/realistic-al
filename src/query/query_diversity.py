from typing import Tuple

import numpy as np
import torch, sys
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy import stats
from torch.utils.data import DataLoader

import vision_transformer as vits
import torchextractor as tx
from kmeans_pytorch import kmeans
import utils
import pickle
import random

from models.bayesian_module import BayesianModule

from .kcenterGreedy import KCenterGreedy
from functools import partial

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

NAMES = ["kcentergreedy", "badge", "freesel"]

DEVICE = "cuda:0"


def query_sampler(
    cfg: DictConfig,
    model: torch.nn.Module,
    labeled_dataloader: DataLoader,
    unlabeled_dataloader: DataLoader,
    acq_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes queries for all diversity based query methods.
    IMPLEMENTATION: Add diversity queries here.

    Args:
        cfg (DictConfig): config
        model (torch.nn.Module): model for queries.
        labeled_dataloader (DataLoader): carries labeled data.
        unlabeled_dataloader (DataLoader): carries unlabeled data.
        acq_size (int): size of query in datapoints.

    Returns:
        Tuple[np.ndarray, np.ndarray]: [pool_indices, ranking values]
    """
    name = cfg.query.name
    ### IMPLEMENTATION
    # Add novel query methods here
    if name == "kcentergreedy":
        indices = _get_kcg(
            model, labeled_dataloader, unlabeled_dataloader, acq_size=acq_size
        )
        # there is no ranking, therefore we add descending numerics as ranking values
        return indices, np.arange(acq_size)[::-1]
    elif name == "badge":
        indices = _get_badge(
            model, labeled_dataloader, unlabeled_dataloader, acq_size=acq_size
        )
        # there is no ranking, therefore we add descending numerics as ranking values
        return indices, np.arange(acq_size)[::-1]
    
    elif name == "freesel":
        indices = _get_freesel(model, labeled_dataloader, unlabeled_dataloader, acq_size=acq_size)
        return indices, np.arange(acq_size)[::-1]
    else:
        raise NotImplementedError


def init_centers(X: np.ndarray, K: int) -> np.ndarray:
    """Determines k centers using KMeans++ of data.
    Source: https://github.com/decile-team/distil/blob/main/distil/active_learning_strategies/badge.py

    Args:
        X (np.ndarray): NxD data
        K (int): #centers to be determined

    Returns:
        np.ndarray: Indices of centers
    """
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.0] * len(X)
    cent = 0
    while len(mu) < K:
        if len(mu) == 1:
            D2 = (
                chunked_pdist(torch.from_numpy(X), torch.from_numpy(mu[-1]))
                .numpy()
                .astype(float)
            )
        else:
            newD = (
                chunked_pdist(torch.from_numpy(X), torch.from_numpy(mu[-1]))
                .numpy()
                .astype(float)
            )
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        D2 = D2.ravel().astype(float)
        Ddist = (D2**2) / sum(D2**2)
        customDist = stats.rv_discrete(name="custm", values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


def chunked_pdist(
    x: torch.Tensor, y: torch.Tensor, device: str = "cuda:0", max_size: int = 8
) -> torch.Tensor:
    """Compute Pairwise Distance for chunks given a maximal size of each chunk part.

    Args:
        x (torch.Tensor): input tensor NxD
        y (torch.Tensor): input tensor MxD
        device (str, optional): device for computation. Defaults to "cuda:0".
        max_size (int, optional): maximum size of chunk of x for pdist computation in GB. Defaults to 8.

    Returns:
        torch.Tensor: NxM pairwise distance matrix
    """
    pdist = torch.nn.PairwiseDistance(p=2)
    new_shape = x.shape[0]
    while (x.element_size() * x.nelement() / 10e9) * (
        new_shape / x.shape[0]
    ) > max_size:
        new_shape /= 2

    y_dev = y.to(device)
    ind_start = 0
    new_shape = x.shape[0] / 2
    dists = []
    while ind_start < x.shape[0]:
        ind_offset = int(min(new_shape, x.shape[0] - ind_start))
        dist_part = pdist(x[ind_start : ind_start + ind_offset].to(device), y_dev).to(
            "cpu"
        )
        dists.append(dist_part)
        ind_start += ind_offset
    dists = torch.concat(dists)
    assert dists.shape[0] == x.shape[0]
    return dists


def get_grad_embedding(
    model: torch.nn.Module,
    dataloader: DataLoader,
    grad_embedding_type: str = "linear",
    device: str = "cuda:0",
) -> torch.Tensor:
    """Compute gradient embedding of final layer of model for each input in dataloader.

    Args:
        model (torch.nn.Module): requires get_features and classifer
        dataloader (DataLoader): contains data.
        grad_embedding_type (str, optional): in [bias, linear,...]. Defaults to "linear".
        device (str, optional): device for gradient computation. Defaults to "cuda:0".

    Returns:
        torch.Tensor: Embedding of last layer gradient
    """

    BayesianModule.k = 1
    start_index = 0
    assert hasattr(model, "model")  # model requires sub class model (due to code)
    assert hasattr(
        model.model, "classifier"
    )  # model.model requires classification head (due to code).
    for i, (x, y) in enumerate(dataloader):
        inputs = x.to(device)
        with torch.no_grad():
            features = model.get_features(inputs)  # B x Z
        l1 = features  # B x Z
        if not model.hparams.model.small_head:
            l1 = model.model.classifier[:-1](features)
        embDim = l1.shape[-1]
        outputs = model.model.classifier(features)

        preds = torch.argmax(outputs, dim=1)

        loss = F.cross_entropy(outputs, preds, reduction="sum")
        l0_grads = torch.autograd.grad(loss, outputs)[0]  # B x C

        # Calculate the linear layer gradients as well if needed
        if grad_embedding_type != "bias":
            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            l1_grads = l0_expand * l1.repeat(1, outputs.shape[-1])  # B x C*Z

        # Populate embedding tensor according to the supplied argument.
        if grad_embedding_type == "bias":
            gradient_embedding = l0_grads
        elif grad_embedding_type == "linear":
            gradient_embedding = l1_grads
        else:
            gradient_embedding = torch.cat([l0_grads, l1_grads], dim=1)
        if i == 0:
            gradient_embeddings = torch.empty(
                [len(dataloader.dataset), gradient_embedding.shape[1]], device="cpu"
            )
        gradient_embeddings[
            start_index : start_index + gradient_embedding.shape[0]
        ] = gradient_embedding.to("cpu")
        start_index += gradient_embedding.shape[0]
        torch.cuda.empty_cache()
    return gradient_embeddings


def _get_badge(
    model: torch.nn.Module,
    labeled_dataloader: DataLoader,
    pool_loader: DataLoader,
    acq_size: int = 100,
):
    assert hasattr(model, "get_features")  # model requires function get_features
    grad_embedding = get_grad_embedding(model, pool_loader)
    grad_embedding = grad_embedding.numpy()
    acq_indices = init_centers(grad_embedding, acq_size)
    return np.array(acq_indices)


def _get_kcg(
    model: torch.nn.Module,
    labeled_dataloader: DataLoader,
    pool_loader: DataLoader,
    acq_size: int = 100,
):
    """Returns the indices of the core-set for the pool of the model via the k-center Greedy approach."""
    assert hasattr(model, "get_features")  # model requires function get_features
    with torch.no_grad():
        features = torch.tensor([]).to(DEVICE)
        for inputs, _ in labeled_dataloader:
            inputs = inputs.to(DEVICE)
            features_batch = model.get_features(inputs)
            features = torch.cat((features, features_batch), 0)
        feat_labeled = features.detach().cpu().numpy()

        features = torch.tensor([]).to(DEVICE)
        for inputs, _ in pool_loader:
            inputs = inputs.to(DEVICE)
            features_batch = model.get_features(inputs)
            features = torch.cat((features, features_batch), 0)
        feat_unlabeled = features.detach().cpu().numpy()

        feat_merge = np.concatenate([feat_labeled, feat_unlabeled], axis=0)
        indices_labeled = np.arange(feat_labeled.shape[0])
        del features
        del feat_labeled
        del feat_unlabeled
        sampling = KCenterGreedy(feat_merge)
        acq_indices = np.array(sampling.select_batch_(indices_labeled, acq_size))

        # subtract the indices of the labeled data to get pool indices
        acq_indices -= indices_labeled.shape[0]
    return acq_indices

def filter_features(dense_features, args, attn=None):
    # input: (n, c, k, k)
    # output: list n: [c1, c2, ...,]
    filtered_features = []
    count = 0

    bs = dense_features.shape[0]
    if "vit" not in args.arch:
        dense_features = dense_features.permute(0, 2, 3, 1)
        dense_features = dense_features.reshape(bs, dense_features.shape[1]*dense_features.shape[2], dense_features.shape[3])  # (n, k*k,c )

    dense_features_norm = torch.norm(dense_features, p=2, dim=2)  # (n, k*k)

    if attn is None:
        mask = dense_features_norm > args.threshold # (n, k*k)
    else:
        assert 0 <= args.threshold <= 1
        # attn: (bs, wh)
        attn_sort, idx_sort = torch.sort(attn, dim=1, descending=False)
        attn_cum = torch.cumsum(attn_sort, dim=1)  # (bs, wh)
        mask = attn_cum > (1-args.threshold)
        for b in range(bs):
            mask[b][idx_sort[b]] = mask[b].clone()

    for b in range(bs):
        mask_i = mask[b]  # (k*k, )
        dense_features_i = dense_features[b]  # (k*k, c)
        if torch.sum(mask_i) > 0:
            dense_features_i = dense_features_i[mask_i]
        else:
            max_id = torch.max(dense_features_norm[b], dim=0)[1]
            dense_features_i = dense_features_i[max_id].unsqueeze(0)  # (1, c)

        if args.centroid_num is not None and args.centroid_num < dense_features_i.shape[0]:
            if args.centroid_num > 1:
                cluster_ids_x, cluster_centers = kmeans(
                    X=dense_features_i, num_clusters=args.sample_num, distance=args.kmeans_dist_type, iter_limit=100, device=torch.device('cuda:0')
                )
            else:
                if args.kmeans_dist_type == "cosine":
                    dense_features_i_ = F.normalize(dense_features_i, p=2, dim=1)
                else:
                    dense_features_i_ = dense_features_i
                cluster_centers = torch.mean(dense_features_i_, dim=0, keepdims=True)
            count += cluster_centers.shape[0]
            filtered_features.append(cluster_centers.cuda())
        else:
            filtered_features.append(dense_features_i)
            count += dense_features_i.shape[0]

    return filtered_features, count

@torch.no_grad()
def extract_features(model, data_loader, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    if "vit" not in args.arch:
        model = tx.Extractor(model, ["layer4"])

    train_ids = []
    train_features = []
    feature_num = 0

    for samples, targets, index in metric_logger.log_every(data_loader, 1):
        samples = samples.cuda(non_blocking=True)
        if "vit" not in args.arch:
            feats, dense_feats = model(samples)
            dense_feats = dense_feats["layer4"].clone()
            dense_feats, count = filter_features(dense_feats, args)
        else:
            dense_feats = model.get_intermediate_layers(samples, n=2)
            dense_feats = dense_feats[0]
            dense_feats = dense_feats[:, 1:]
            attn = model.get_last_selfattention(samples)  # (bs, nh, wh+1, wh+1)
            attn = torch.mean(attn, dim=1)[:, 0, 1:]  # (bs, wh)
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            dense_feats, count = filter_features(dense_feats, args, attn)

        feature_num += count

        train_features.extend(dense_feats)
        train_ids.extend(index)

    return train_features, train_ids

def merge_features(all_features):
    merged_features = list(all_features.values())
    merged_features = torch.cat(merged_features, dim=0)
    id2idx = {}
    idx = 0
    count = 0
    merged_ids = []
    for id in all_features:
        id2idx[count] = torch.arange(idx, idx+all_features[id].shape[0])
        merged_ids.append(id)
        idx = idx + all_features[id].shape[0]
        count += 1
    return merged_features, merged_ids, id2idx

def _get_freesel(
    model: torch.nn.Module,
    labeled_dataloader: DataLoader,
    pool_loader: DataLoader,
    acq_size: int = 100,
):
    print("We have come to freesel")
    print("Model: ", model)
    print("Labeled Dataloader: ", len(labeled_dataloader))
    print("Unlabeled Dataloader: ", len(pool_loader))
    print("Acquisition Size: ", acq_size)

    PARAMS = {
        "arch": "vit_small",
        "patch_size": 16,
        'threshold': 0.5,
        "centroid_num": 1,
        "kmeans_dist_type": "euclidean",
        "sample_num": 5,
        "random_num": None,
        "sampling": "prob",
        "selected_num": acq_size,
        "dist_type": "cosine",
        "pretrained_weights": '',
        'checkpoint_key': 'teacher'

    }

    PARAMS = DictToObject(PARAMS)

    #1. Load in a different model.
    model = vits.__dict__[PARAMS.arch](patch_size=PARAMS.patch_size, num_classes=0)
    model.cuda()
    utils.load_pretrained_weights(model, PARAMS.pretrained_weights, PARAMS.checkpoint_key, PARAMS.arch, PARAMS.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features, train_ids = extract_features(model, pool_loader, PARAMS)

    all_features = {}
    for i in range(len(train_features)):
        all_features[train_ids[i]] = train_features[i]

    merged_features, merged_ids, id2idx = merge_features(all_features)

    print("Select Samples...")

    args = PARAMS

    if args.random_num is None:
        if args.sampling == "prob":
            selected_sample = utils.prob_seed_dense(merged_features, id2idx, args.selected_num, partial(utils.get_distance, type=args.dist_type))
        else:
            selected_sample = utils.farthest_distance_sample_dense(merged_features, id2idx, args.selected_num, partial(utils.get_distance, type=args.dist_type))
    else:
        init_ids = random.sample(range(len(id2idx)), args.random_num)
        if args.sampling == "prob":
            selected_sample = utils.prob_seed_dense(merged_features, id2idx, args.selected_num, partial(utils.get_distance, type=args.dist_type), init_ids=init_ids)
        else:
            selected_sample = utils.farthest_distance_sample_dense(merged_features, id2idx, args.selected_num, partial(utils.get_distance, type=args.dist_type), init_ids=init_ids)

    selected_ids = []
    for idx in selected_sample:
        id = merged_ids[idx]
        selected_ids.append(int(id))
    # selected_ids.sort()

    return selected_ids