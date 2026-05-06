import argparse
import math
import os
import pickle
import sys
import __main__
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image, ImageFilter
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import Tokenizer
from models import Encoder, Decoder


def load_tokenizer(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate(decoder, features, tokenizer, max_length=20):
    decoder.eval()
    device = features.device

    start_idx = tokenizer.word_index["<start>"]
    end_idx = tokenizer.word_index["<end>"]

    h, c = decoder.init_hidden_state(features)
    word = torch.tensor([start_idx], dtype=torch.long, device=device)

    words = []
    alphas = []

    with torch.no_grad():
        for _ in range(max_length):
            embedding = decoder.embedding(word)

            alpha = decoder.attention(features, h)
            alphas.append(alpha.squeeze(0).detach().cpu().numpy())

            if decoder.hard_attention:
                sampled_idx = alpha.argmax(dim=1)
                context = features[
                    torch.arange(features.size(0), device=device), sampled_idx
                ]
            else:
                context = (alpha.unsqueeze(2) * features).sum(dim=1)
                beta = torch.sigmoid(decoder.f_beta(h))
                context = beta * context

            lstm_input = torch.cat([embedding, context], dim=1)
            h, c = decoder.decode_step(lstm_input, (h, c))

            scores = decoder.L_o(decoder.L_h(h) + decoder.L_z(context) + embedding)
            next_word = scores.argmax(dim=1)
            word_idx = next_word.item()

            if word_idx == end_idx:
                break

            words.append(tokenizer.index_word.get(word_idx, "UNK"))
            word = next_word

    return words, alphas


def alpha_to_map(alpha, image_size, hard_attention=False):
    width, height = image_size
    alpha = np.asarray(alpha, dtype=np.float32)

    side = int(np.sqrt(alpha.size))
    alpha = alpha.reshape(side, side)

    if hard_attention:
        hard = np.zeros_like(alpha)
        row, col = np.unravel_index(alpha.argmax(), alpha.shape)
        hard[row, col] = 1.0

        mask = Image.fromarray((hard * 255).astype(np.uint8), mode="L")
        mask = mask.resize((width, height), Image.NEAREST)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=5))

        arr = np.asarray(mask, dtype=np.float32) / 255.0
        return arr / (arr.max() + 1e-8)

    alpha -= alpha.min()
    alpha /= alpha.max() + 1e-8

    mask = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    mask = mask.resize((width, height), Image.BILINEAR)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=10))

    arr = np.asarray(mask, dtype=np.float32) / 255.0
    arr -= arr.min()
    return arr / (arr.max() + 1e-8)


def make_sat_panel(image, alpha, hard_attention=False):
    image = image.convert("RGB")
    img = np.asarray(image, dtype=np.float32) / 255.0
    height, width = img.shape[:2]

    att = alpha_to_map(alpha, (width, height), hard_attention)

    dark = img * 0.28
    white = np.ones_like(img)

    strength = 0.90 if hard_attention else 0.78
    out = dark * (1.0 - att[:, :, None] * strength) + white * (
        att[:, :, None] * strength
    )

    return np.clip(out, 0, 1)


def make_output_path(image_path, checkpoint_path, output_dir):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]

    folder = os.path.join(output_dir, image_name)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{checkpoint_name}.pdf")


def save_attention_pdf(
    image, words, alphas, checkpoint_path, output_path, hard_attention=False
):
    n_panels = len(words) + 1
    cols = 5
    rows = math.ceil(n_panels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 2.8 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    axes = axes.flatten()

    axes[0].imshow(image)
    axes[0].axis("off")

    for i, word in enumerate(words):
        ax = axes[i + 1]
        panel = make_sat_panel(image, alphas[i], hard_attention)

        ax.imshow(panel)
        ax.set_title(word, fontsize=9, fontweight="bold")
        ax.axis("off")

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.97,
        bottom=0.03,
        wspace=0.25,
        hspace=0.35,
    )

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")

    plt.close(fig)
    print(f"Saved PDF to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/flickr8k/processed")
    parser.add_argument("--feature_extractor", default="vgg", choices=["vgg", "resnet"])
    parser.add_argument("--hard_attention", action="store_true")
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--output_dir", default="results")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = load_tokenizer(os.path.join(args.data_dir, "tokenizer.pkl"))

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

    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "model_state_dict" in checkpoint:
        decoder.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "decoder_state_dict" in checkpoint:
        decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=False)
    else:
        decoder.load_state_dict(checkpoint, strict=False)

    decoder.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    display_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )

    raw_image = Image.open(args.image_path).convert("RGB")
    display_image = display_transform(raw_image)
    image_tensor = preprocess(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image_tensor)
        words, alphas = evaluate(decoder, features, tokenizer, args.max_length)

    print("\nGenerated caption:")
    print(" ".join(words))
    print()

    output_path = make_output_path(
        args.image_path,
        args.checkpoint,
        args.output_dir,
    )

    save_attention_pdf(
        image=display_image,
        words=words,
        alphas=alphas,
        checkpoint_path=args.checkpoint,
        output_path=output_path,
        hard_attention=args.hard_attention,
    )


if __name__ == "__main__":
    main()
