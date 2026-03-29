import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score


class EarlyStopping:
    """Early stopping based on BLEU score"""

    def __init__(
        self, patience=5, delta=0, verbose=False, path="checkpoints/best_model.pth"
    ):
        self.patience = patience
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = 0.0
        self.early_stop = False
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def check_early_stop(self, bleu, model, optimizer, epoch):
        if bleu > self.best_score:
            self.best_score = bleu
            if self.verbose:
                print(
                    f"Validation BLEU increased ({self.best_score:.4f} -> {bleu:.4f}). Saving model."
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_bleu": self.best_score,
                },
                self.path,
            )

            self.counter = 0

        else:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True


def cross_entropy(
    preds,
    targets,
    decode_lengths,
    pad_idx=0,
    reduce_mean=True,
):
    """If reduce_mean is False, returns per-sample average loss for REINFORCE."""
    B, T, V = preds.size()
    device = preds.device

    preds_flat = preds.view(-1, V)
    targets_flat = targets.contiguous().view(-1)

    if reduce_mean:
        # soft attention
        return F.cross_entropy(
            preds_flat, targets_flat, ignore_index=pad_idx, reduction="mean"
        )
    else:
        # hard attention
        loss = F.cross_entropy(preds, targets, ignore_index=pad_idx, reduction="none")
        loss = loss.view(B, T)
        mask = torch.arange(T, device=device).unsqueeze(0) < decode_lengths.unsqueeze(1)
        loss = loss * mask
        return loss.sum(dim=1) / decode_lengths.float()


def soft_attn_loss(preds, targets, decode_lengths, alphas, lambda_reg=1.0, pad_idx=0):
    """Deterministic "Soft" Attention loss with doubly stochastic regularization."""
    ce_loss = cross_entropy(preds, targets, decode_lengths, pad_idx)
    reg = ((1.0 - alphas.sum(dim=1)) ** 2).sum(dim=1).mean()
    total_loss = ce_loss + lambda_reg * reg
    return total_loss, ce_loss, reg


def hard_attn_loss(
    preds, targets, decode_lengths, alphas, log_probs, baseline=None, pad_idx=0
):
    """Hard attention loss."""
    device = preds.device
    B, T = log_probs.size()

    ce_loss_per_sample = cross_entropy(
        preds, targets, decode_lengths, pad_idx, reduce_mean=False
    )

    reward = -ce_loss_per_sample.detach()

    avg_reward = reward.mean()
    if baseline is None:
        baseline = avg_reward
    else:
        baseline = 0.9 * baseline + 0.1 * avg_reward

    advantage = reward - baseline

    mask = torch.arange(T, device=device).unsqueeze(0) < decode_lengths.unsqueeze(1)
    log_probs = log_probs * mask

    seq_log_probs = log_probs.sum(dim=1) / decode_lengths.float()
    reinforce_loss = -(seq_log_probs * advantage).mean()

    entropy = -(alphas * torch.log(alphas.clamp(min=1e-8))).sum(dim=2)
    entropy = (entropy * mask).sum(dim=1) / decode_lengths.float()

    total_loss = ce_loss_per_sample.mean() + reinforce_loss - 0.01 * entropy.mean()

    return total_loss, ce_loss_per_sample.mean(), reinforce_loss, baseline


def train_and_validate(
    decoder,
    train_loader,
    val_loader,
    device,
    num_epochs=50,
    patience=5,
    lr=1e-3,
    alpha_c=1.0,
    resume_ckpt=None,
):
    optimizer = torch.optim.RMSprop(decoder.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    start_epoch = 0
    if resume_ckpt and os.path.exists(resume_ckpt):
        checkpoint = torch.load(resume_ckpt)
        decoder.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(
            f"Resumed training from epoch {start_epoch} with validation loss {checkpoint['val_loss']:.6}"
        )

    for epoch in range(start_epoch, num_epochs):
        decoder.train()

    criterion = nn.CrossEntropyLoss().to(device)
