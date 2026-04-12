"""
Usage: python code/train.py --data_dir data/flickr8k/processed --ckpt_path checkpoints/sat_best.pth --epochs 30 --batch_size 64 --lr 4e-4 --decoding greedy
"""

import argparse, pickle, sys, os, time
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

from models import Decoder
from dataset import FeatureCaptionDataset, Tokenizer

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)


class EarlyStopping:
    """Early stopping based on BLEU score."""

    def __init__(self, patience=5, verbose=True, path="checkpoints/best_model.pth"):
        self.patience = patience
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = 0.0
        self.early_stop = False
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def check_early_stop(
        self,
        val_bleu1,
        val_bleu2,
        val_bleu3,
        val_bleu4,
        val_meteor,
        model,
        optimizer,
        epoch,
    ):
        if val_bleu4 > self.best_score:
            self.best_score = val_bleu4
            if self.verbose:
                print(
                    f"Validation BLEU increased ({self.best_score:.4f} -> {val_bleu4:.4f}). Saving model."
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_bleu": self.best_score,
                    "val_bleu1": val_bleu1,
                    "val_bleu2": val_bleu2,
                    "val_bleu3": val_bleu3,
                    "val_bleu4": val_bleu4,
                    "val_meteor": val_meteor,
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
        preds_flat = preds.view(-1, V)
        targets_flat = targets.contiguous().view(-1)

        loss = F.cross_entropy(
          preds_flat,
          targets_flat,
          ignore_index=pad_idx,
          reduction="none"
        )

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

    reward = -ce_loss_per_sample
    reward = reward.detach()

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

    total_loss = ce_loss_per_sample.mean() + reinforce_loss - 0.005 * entropy.mean()

    return total_loss, ce_loss_per_sample.mean(), reinforce_loss, baseline


def decode_sequence(indices, tokenizer):
    """Convert a sequence of token indices to a caption string."""
    words = []
    for i in indices:
        word = tokenizer.index_word.get(i, "UNK")
        if word == "<end>" or word == "PAD":
            break
        if word != "<start>":
            words.append(word)

    return words


def generate_caption_greedy(decoder, features, tokenizer, max_length=20):
    decoder.eval()
    batch_size = features.size(0)
    device = features.device

    start_idx = tokenizer.word_index["<start>"]
    end_idx = tokenizer.word_index["<end>"]

    h, c = decoder.init_hidden_state(features)
    prev_words = torch.full((batch_size,), start_idx, dtype=torch.long).to(device)

    finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
    sequences = [[] for _ in range(batch_size)]

    for _ in range(max_length):
        embeddings = decoder.embedding(prev_words)
        alpha = decoder.attention(features, h)

        if not decoder.hard_attention:
            context = (alpha.unsqueeze(2) * features).sum(dim=1)
            beta = torch.sigmoid(decoder.f_beta(h))
            context = beta * context

        else:
            _, sampled_idx = alpha.max(dim=1)
            batch_idx = torch.arange(batch_size).to(device)
            context = features[batch_idx, sampled_idx, :]

        lstm_input = torch.cat([embeddings, context], dim=1)
        h, c = decoder.decode_step(lstm_input, (h, c))
        preds = decoder.L_o(decoder.L_h(h) + decoder.L_z(context) + embeddings)

        _, next_words = preds.max(dim=1)
        for i in range(batch_size):
            if not finished[i]:
                word_idx = next_words[i].item()
                sequences[i].append(word_idx)
                if word_idx == end_idx:
                    finished[i] = True

        if finished.all():
            break

        prev_words = next_words

    return sequences


def generate_caption_beam_search(
    decoder, encoder_out, tokenizer, beam_size=3, max_len=20
):
    decoder.eval()
    k = beam_size
    vocab_size = decoder.vocab_size
    device = encoder_out.device

    start_idx = tokenizer.word_index["<start>"]
    end_idx = tokenizer.word_index["<end>"]

    h, c = decoder.init_hidden_state(encoder_out)

    encoder_out = encoder_out.expand(k, -1, -1)
    h = h.expand(k, -1)
    c = c.expand(k, -1)

    k_prev_words = torch.full((k, 1), start_idx, dtype=torch.long, device=device)
    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1, device=device)

    complete_seqs = []
    complete_seqs_scores = []

    step = 1
    while True:
        emb = decoder.embedding(k_prev_words.squeeze(1))
        alpha = decoder.attention(encoder_out[:k], h)

        if not decoder.hard_attention:
            context = (alpha.unsqueeze(2) * encoder_out[:k]).sum(dim=1)
            beta = torch.sigmoid(decoder.f_beta(h))
            context = beta * context
        else:
            idx = alpha.argmax(dim=1)
            batch_idx = torch.arange(k, device=device)
            context = encoder_out[:k][batch_idx, idx]

        h, c = decoder.decode_step(torch.cat([emb, context], dim=1), (h, c))

        preds = decoder.L_o(decoder.L_h(h) + decoder.L_z(context) + emb)
        log_probs = F.log_softmax(preds, dim=1)

        log_probs = top_k_scores.expand_as(log_probs) + log_probs

        if step == 1:
            top_k_scores, top_k_words = log_probs[0].topk(
                k, dim=0, largest=True, sorted=True
            )
        else:
            top_k_scores, top_k_words = log_probs.view(-1).topk(
                k, dim=0, largest=True, sorted=True
            )

        prev_word_inds = top_k_words // vocab_size
        next_word_inds = top_k_words % vocab_size

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

        incomplete_inds = [
            ind for ind, next_word in enumerate(next_word_inds) if next_word != end_idx
        ]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            length_norm = step**0.7
            scores = top_k_scores[complete_inds].view(-1) / length_norm
            complete_seqs_scores.extend(scores.tolist())

        k -= len(complete_inds)
        if k == 0:
            break

        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[:k]
        top_k_scores = top_k_scores[incomplete_inds].view(-1, 1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        step += 1
        if step > max_len:
            break

    if not complete_seqs:
        complete_seqs = seqs.tolist()
        complete_seqs_scores = (top_k_scores.view(-1) / step).tolist()

    best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
    best_seq = complete_seqs[best_idx]

    return best_seq


def print_best_checkpoint_metrics(path):
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return

    checkpoint = torch.load(path, map_location="cpu", weights_only = False)
    print("\n=== Best Checkpoint Metrics ===")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"BLEU-1: {checkpoint['val_bleu1']:.4f}")
    print(f"BLEU-2: {checkpoint['val_bleu2']:.4f}")
    print(f"BLEU-3: {checkpoint['val_bleu3']:.4f}")
    print(f"BLEU-4: {checkpoint['val_bleu4']:.4f}")
    print(f"METEOR: {checkpoint['val_meteor']:.4f}")


def train_and_validate(
    decoder,
    train_loader,
    val_loader,
    val_ref_dict,
    device,
    tokenizer,
    configs,
    resume_ckpt=None,
):
    optimizer = torch.optim.RMSprop(decoder.parameters(), lr=configs["lr"])
    early_stopping = EarlyStopping(
        patience=configs["patience"], path=configs["checkpoint_path"]
    )

    baseline = None
    pad_idx = tokenizer.word_index["PAD"]

    decoding_strategy = configs.get("decoding_strategy", "greedy")

    start_epoch = 0
    if resume_ckpt and os.path.exists(resume_ckpt):
        checkpoint = torch.load(resume_ckpt)
        decoder.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(
            f"Resumed training from epoch {start_epoch} with BLEU-4 {checkpoint['best_bleu']:.4f}"
        )

    for epoch in range(start_epoch, configs["epochs"] + 1):
        decoder.train()
        train_loss = 0.0
        start_time = time.time()

        for batch_idx, (features, captions, image_ids) in enumerate(train_loader):
            features = features.to(device)
            captions = captions.to(device)

            decode_lengths = (captions != pad_idx).sum(dim=1)
            decode_lengths, sort_ind = decode_lengths.sort(dim=0, descending=True)

            features = features[sort_ind]
            captions = captions[sort_ind]

            preds, _, _, alphas, log_probs = decoder(
                features, captions[:, :-1], decode_lengths
            )
            targets = captions[:, 1:][:, : preds.size(1)]

            if decoder.hard_attention:
                loss, ce_loss, reinforce_loss, baseline = hard_attn_loss(
                    preds, targets, decode_lengths, alphas, log_probs, baseline, pad_idx
                )
            else:
                loss, ce_loss, reg_loss = soft_attn_loss(
                    preds,
                    targets,
                    decode_lengths,
                    alphas,
                    configs["lambda_reg"],
                    pad_idx,
                )

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % configs["log_interval"] == 0:
                print(
                    f"Epoch [{epoch}/{configs['epochs']}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / (batch_idx + 1)

        decoder.eval()
        references = []
        hypotheses = []

        evaluated_images = set()

        with torch.no_grad():
            for features, captions, image_ids in val_loader:
                features = features.to(device)

                batch_size = features.size(0)
                indices_eval = [
                    i for i in range(batch_size) if image_ids[i] not in evaluated_images
                ]

                if not indices_eval:
                    continue

                if decoding_strategy == "greedy":
                    features_eval = features[indices_eval]
                    img_ids_eval = [image_ids[i] for i in indices_eval]

                    seqs = generate_caption_greedy(decoder, features_eval, tokenizer)

                    for i, img_id in enumerate(img_ids_eval):
                        evaluated_images.add(img_id)
                        hyp_words = decode_sequence(seqs[i], tokenizer)
                        ref_word_list = val_ref_dict[img_id]

                        hypotheses.append(hyp_words)
                        references.append(ref_word_list)

                elif decoding_strategy == "beam":
                    for i in indices_eval:
                        img_id = image_ids[i]
                        evaluated_images.add(img_id)

                        single_feature = features[i].unsqueeze(0)
                        beam_size = configs.get("beam_size", 3)

                        best_seq = generate_caption_beam_search(
                            decoder, single_feature, tokenizer, beam_size=beam_size
                        )

                        hyp_words = decode_sequence(best_seq, tokenizer)
                        ref_word_list = val_ref_dict[img_id]

                        hypotheses.append(hyp_words)
                        references.append(ref_word_list)

            val_bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
            val_bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
            val_bleu3 = corpus_bleu(
                references, hypotheses, weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0)
            )
            val_bleu4 = corpus_bleu(
                references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)
            )

            meteor_scores = []
            for ref_list, hyp_words in zip(references, hypotheses):
                hyp_sentence = " ".join(hyp_words)
                ref_sentences = [" ".join(ref) for ref in ref_list]
                meteor_scores.append(
                    meteor_score(
                        [r.split() for r in ref_sentences], hyp_sentence.split()
                    )
                )

            val_meteor = np.mean(meteor_scores)

            epoch_time = time.time() - start_time
            print(
                "Epoch [{}/{}] completed in {:.2f} seconds".format(
                    epoch, configs["epochs"], epoch_time
                )
            )
            print("Train loss: {:.4f}".format(avg_train_loss))
            print(
                "Validation BLEU-1: {:.4f}, BLEU-2: {:.4f}, BLEU-3: {:.4f}, BLEU-4: {:.4f}, METEOR: {:.4f}".format(
                    val_bleu1, val_bleu2, val_bleu3, val_bleu4, val_meteor
                )
            )

            early_stopping.check_early_stop(
                val_bleu1,
                val_bleu2,
                val_bleu3,
                val_bleu4,
                val_meteor,
                decoder,
                optimizer,
                epoch,
            )

            if early_stopping.early_stop:
                print(f"Early stopping after {epoch} epochs without improvement.")
                break


def parse_args():
    parser = argparse.ArgumentParser(description="Show Attend Tell")

    # Data/paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/flickr8k/processed",
        help="Path to processed data",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/sat_best.pth",
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--feature_extractor",
        type=str,
        default="vgg",
        choices=["vgg", "resnet"],
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-4,
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=1.0,
        help="Regularization weight for doubly stochastic attention",
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, help="Print loss every N batches"
    )

    # Evaluation Configs
    parser.add_argument(
        "--decoding",
        type=str,
        default="greedy",
        choices=["greedy", "beam"],
        help="Inference strategy",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=3,
        help="Beam size if using beam search decoding",
    )

    # Hard attention
    parser.add_argument(
        "--hard_attention",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configs = {
        "epochs": args.epochs,
        "lr": args.lr,
        "patience": args.patience,
        "checkpoint_path": args.ckpt_path,
        "lambda_reg": args.lambda_reg,
        "log_interval": args.log_interval,
        "decoding_strategy": args.decoding,
        "beam_size": args.beam_size,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load preprocessed data
    print(f"Loading tokenizer and datasets from {args.data_dir}...")
    with open(os.path.join(args.data_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    train_df = pd.read_csv(os.path.join(args.data_dir, "train_df.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val_df.csv"))

    feature_name = args.feature_extractor

    with open(os.path.join(args.data_dir, f"train_{feature_name}.pkl"), "rb") as f:
        train_features = pickle.load(f)
    with open(os.path.join(args.data_dir, f"val_{feature_name}.pkl"), "rb") as f:
        val_features = pickle.load(f)
    train_cap_vector = np.load(os.path.join(args.data_dir, "train_cap_vector.npy"))
    val_cap_vector = np.load(os.path.join(args.data_dir, "val_cap_vector.npy"))

    train_dataset = FeatureCaptionDataset(train_df, train_cap_vector, train_features)
    val_dataset = FeatureCaptionDataset(val_df, val_cap_vector, val_features)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("Building validation reference dictionary...")
    val_ref_dict = {}
    for _, row in val_df.iterrows():
        img_id = row["Image_ID"]
        cap_clean = row["Caption"].replace("<start>", "").replace("<end>", "").strip()
        if img_id not in val_ref_dict:
            val_ref_dict[img_id] = []
        val_ref_dict[img_id].append(cap_clean.split())

    if args.feature_extractor == "vgg":
        enc_dim = 512
    else:
        enc_dim = 2048

    decoder = Decoder(
        vocab_size=len(tokenizer.word_index),
        embed_dim=256,
        encoder_dim=enc_dim,
        decoder_dim=512,
        attention_dim=512,
        hard_attention=args.hard_attention,
    ).to(device)

    print("Training configuration:")
    for key, value in configs.items():
        print(f"  {key}: {value}")

    print("\nStarting Training!")
    train_and_validate(
        decoder,
        train_loader,
        val_loader,
        val_ref_dict,
        device,
        tokenizer,
        configs,
        resume_ckpt=args.resume,
    )

    print_best_checkpoint_metrics(configs["checkpoint_path"])
