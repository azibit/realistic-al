import math
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils.tensor import to_numpy

from .batchbald_redux.batchbald import get_batchbald_batch

### IMPLEMENTATION
# Add name of uncerainty method here and add computation _get_xxx_function.
NAMES = ["bald", "entropy", "random", "batchbald", "variationratios"]


def get_acq_function(cfg, pt_model) -> Callable[[torch.Tensor], torch.Tensor]:
    name = str(cfg.query.name).split("_")[0]
    if name == "bald":
        return _get_bald_fct(pt_model)
    elif name == "entropy":
        return _get_bay_entropy_fct(pt_model)
    elif name == "random":
        return _get_random_fct()
    elif name == "batchbald":
        return get_bay_logits(pt_model)
    elif name == "variationratios":
        return _get_var_ratios(pt_model)
    else:
        raise NotImplementedError


def get_post_acq_function(
    cfg: DictConfig, device="cuda:0"
) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    names = str(cfg.query.name).split("_")
    if cfg.query.name == "batchbald":
        # This values should only be used to select the entropy computation
        num_samples = 40000  # taken from BatchBALD

        def post_acq_function(logprob_n_k_c: np.ndarray, acq_size: int):
            """BatchBALD acquisition function using logits with iterative conditional mutual information."""
            assert (
                len(logprob_n_k_c.shape) == 3
            )  # make sure that input is of correct type
            logprob_n_k_c = torch.from_numpy(logprob_n_k_c).to(
                device=device, dtype=torch.double
            )
            with torch.no_grad():
                out = get_batchbald_batch(
                    logprob_n_k_c,
                    batch_size=acq_size,
                    num_samples=num_samples,
                    dtype=torch.double,
                    device=device,
                )
            indices = np.array(out.indices)
            scores = np.array(out.scores)
            return indices, scores

        return post_acq_function
    else:

        def post_acq_function(acq_scores: np.ndarray, acq_size: int):
            """Acquires based on ranking. Highest ranks are acquired first."""
            assert len(acq_scores.shape) == 1  # make sure that input is of correct type
            acq_ind = np.arange(len(acq_scores))
            inds = np.argsort(acq_scores)[::-1]
            inds = inds[:acq_size]
            acq_list = acq_scores[inds]
            acq_ind = acq_ind[inds]
            return inds, acq_list

        return post_acq_function


###


def query_sampler(
    dataloader: DataLoader,
    acq_function,
    post_acq_function,
    acq_size: int = 64,
    device="cuda:0",
):
    """Returns the queries (acquistion values and indices) given the data pool and the acquisition function.
    The Acquisition Function Returns Numpy arrays!"""
    acq_list = None
    counts = 0
    for i, batch in enumerate(dataloader):
        acq_values = acq_from_batch(batch, acq_function, device=device)
        if acq_list is None:
            shape = acq_values.shape
            new_shape = (len(dataloader) * dataloader.batch_size, *shape[1:])
            acq_list = np.zeros(new_shape)
        acq_list[counts : counts + len(acq_values)] = acq_values
        counts += len(acq_values)
    acq_list = acq_list[:counts]
    acq_ind, acq_scores = post_acq_function(acq_list, acq_size)

    return acq_ind, acq_scores


def _get_bay_entropy_fct(pt_model: torch.nn.Module):
    def acq_bay_entropy(x: torch.Tensor):
        """Returns the Entropy of predictions of the bayesian model"""
        with torch.no_grad():
            out = pt_model(x, agg=False)  # BxkxD
            ent = pred_entropy(out)
        return ent

    return acq_bay_entropy


def _get_exp_entropy_fct(pt_model: torch.nn.Module):
    def acq_exp_entropy(x: torch.Tensor):
        """Returns the expected entropoy of some probabilistic model."""
        with torch.no_grad():
            out = pt_model(x, agg=False)
            ex_ent = exp_entropy(out)
        return ex_ent

    return acq_exp_entropy


def _get_bald_fct(pt_model: torch.nn.Module):
    def acq_bald(x: torch.Tensor):
        """Returns the BALD-acq values (Mutual Information) between most likely labels and the model parameters"""
        with torch.no_grad():
            out = pt_model(x, agg=False)
            mut_info = mutual_bald(out)
        return mut_info

    return acq_bald


def get_bay_logits(pt_model: torch.nn.Module):
    def acq_logits(x: torch.Tensor):
        """Returns the NxKxC logprobs needed for BatchBALD"""
        with torch.no_grad():
            out = pt_model(x, agg=False)
            out = torch.log_softmax(out, dim=2)
        return out

    return acq_logits


def _get_var_ratios(pt_model: torch.nn.Module):
    def acq_var_ratios(x: torch.Tensor):
        """Returns the variation ratio values."""
        with torch.no_grad():
            out = pt_model(x, agg=False)
            out = var_ratios(out)
        return out

    return acq_var_ratios


def _get_random_fct():
    def acq_random(x: torch.Tensor, c: float = 0.0001):
        """Returns random values over the interval [0, c)"""
        out = torch.rand(x.shape[0], device=x.device) * c
        return out

    return acq_random


def pred_entropy(logits: torch.Tensor):
    """Get the mean entropy of multiple logits."""
    k = logits.shape[1]
    out = F.log_softmax(logits, dim=2)  # BxkxD
    # This part was wrong but it performed better than BatchBALD - interesting
    # out = out.mean(dim=1)  # BxD
    out = torch.logsumexp(out, dim=1) - math.log(k)  # BxkxD --> BxD
    ent = torch.sum(-torch.exp(out) * out, dim=1)  # B
    return ent


def var_ratios(logits: torch.Tensor):
    k = logits.shape[1]
    out = F.log_softmax(logits, dim=2)  # BxkxD
    out = torch.logsumexp(out, dim=1) - math.log(k)  # BxkxD --> BxD
    out = 1 - torch.exp(out.max(dim=-1).values)  # B
    return out


def exp_entropy(logits: torch.Tensor):
    out = F.log_softmax(logits, dim=2)  # BxkxD
    out = torch.sum(-torch.exp(out) * out, dim=2)  # Bxk
    out = torch.mean(out, dim=1)
    return out


def mutual_bald(logits: torch.Tensor):
    return pred_entropy(logits) - exp_entropy(logits)


def acq_from_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    function: Callable[[torch.Tensor], torch.Tensor],
    device="cuda:0",
) -> np.ndarray:
    """Compute function from batch inputs.

    Args:
        batch (Tuple[torch.Tensor, torch.Tensor]): [inputs, labels]
        function (Callable[[torch.Tensor], torch.Tensor]): function where outputs are desired.
        device (str, optional): device for computation. Defaults to "cuda:0".

    Returns:
        np.ndarray: outputs of function for batch inputs.
    """
    x, y, index = batch
    x = x.to(device)
    out = function(x)
    out = to_numpy(out)
    return out
