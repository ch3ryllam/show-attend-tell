"""
Usage: python code/visualize.py --image_path {path_to_imag.jpg} --checkpoint checkpoints/sat_best.pth

Assumes data_dir is data/flickr8k/processed
"""

import argparse
import pickle
import os
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models import Encoder, Decoder
from dataset import Tokenizer


def evaluate(decoder, features, tokenizer, max_length=20):
    """Generates a cpation and returns the attention weights for each word."""
    decoder.eval()
    device = features.device

    start_idx = tokenizer.word_index["<start>"]
    end_idx = tokenizer.word_index["<end>"]

    h, c = decoder.init_hidden_state(features)

    word = torch.tensor([start_idx], dtype=torch.long).to(device)

    seq = []
    alphas = []

    with torch.no_grad():
        for _ in range(max_length):
            embeddings = decoder.embedding(word)

            alpha = decoder.attention(features, h)
            alphas.append(alpha.squeeze(0).cpu().numpy())

            if not decoder.hard_attention:
                # soft attention
                context = (alpha.unsqueeze(2) * features).sum(dim=1)
                beta = torch.sigmoid(decoder.f_beta(h))
                context = beta * context

            else:
                # hard attention
                _, sampled_idx = alpha.max(dim=1)
                context = features[0, sampled_idx, :]

            lstm_input = torch.cat([embeddings, context], dim=1)
            h, c = decoder.decode_step(lstm_input, (h, c))
            preds = decoder.L_o(decoder.L_h(h) + decoder.L_z(context) + embeddings)

            _, next_word = preds.max(dim=1)
            word_idx = next_word.item()

            if word_idx == end_idx:
                break

            seq.append(word_idx)
            word = next_word

    # decode sequence
    words = [tokenizer.index_word.get(i, "UNK") for i in seq]
    return words, alphas


def plot_attention(image_path, words, alphas):
    """Plots the image with attention weights overlaid for each generated word"""

    image = Image.open(image_path).convert("RGB")
    num_words = len(words)
    cols = 4
    rows = math.ceil(num_words / cols)

    fig = plt.figure(figsize=(15, 3 * rows))

    for t in range(num_words):
        ax = fig.add_subplot(rows, cols, t + 1)

        ax.set_title(words[t], fontsize=14, backgroundcolor="white", loc="left")

        # Plot original image first
        ax.imshow(image)

        # 14 x 14 image
        alpha_img = alphas[t].reshape(14, 14)
        alpha_img = Image.fromarray(alpha_img)
        alpha_img = alpha_img.resize(image.size, Image.LANCZOS)

        alpha_arr = np.array(alpha_img)
        alpha_arr = (alpha_arr - alpha_arr.min()) / (
            alpha_arr.max() - alpha_arr.min() + 1e-8
        )

        # Create black image with alpha transparency
        img_width, img_height = image.size
        black_overlay = np.zeros((img_height, img_width, 4))

        # Set transparency of black overlay using inverse of attention weights
        # Multiply by 0.9 so darker areas aren't completely black
        black_overlay[:, :, 3] = 1 - alpha_arr * 0.9

        # Draw black overlay on top of original image
        ax.imshow(black_overlay)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize attention")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/sat_soft_vgg.pth"
    )
    parser.add_argument("--data_dir", type=str, default="data/flickr8k/processed")
    parser.add_argument(
        "--feature_extractor", type=str, default="vgg", choices=["vgg", "resnet"]
    )
    parser.add_argument("--hard_attention", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    with open(os.path.join(args.data_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
        print("Loaded tokenizer")

    # intialize encoder + decoder
    encoder = Encoder(model_type=args.feature_extractor).to(device)
    encoder.eval()

    decoder = Decoder(
        vocab_size=len(tokenizer.word_index),
        embed_dim=256,
        encoder_dim=encoder.enc_dim,
        decoder_dim=512,
        attention_dim=512,
        hard_attention=args.hard_attention,
    ).to(device)
    print("Loaded encoder/decoder")

    # load trained weights
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    decoder.load_state_dict(checkpoint["model_state_dict"])

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print("Loaded checkpoint")

    print(f"Processing image: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image_tensor)
        words, alphas = evaluate(decoder, features, tokenizer)

    caption = " ".join(words)
    print(f"\nGenerated Caption: {caption}\n")

    # Plot
    print("Generating visualization...")
    plot_attention(args.image_path, words, alphas)
