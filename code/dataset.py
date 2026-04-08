import os
import re
import pickle
import numpy as np
import pandas as pd
import torch

from PIL import Image
from collections import Counter
from sklearn.utils import shuffle
from torchvision import models, transforms
from torchvision.models import VGG16_Weights, ResNet50_Weights

from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_sentence(sentence):
    return list(filter(lambda x: len(x) > 0, re.split(r"\W+", sentence.lower())))


class Tokenizer:
    def __init__(
        self, num_words=None, oov_token="UNK", filters='!"#$%&()*+.,-/:;=?@[\\]^_`{|}~'
    ):
        self.num_words = num_words
        self.oov_token = oov_token
        self.filters = filters
        self.word_counts = Counter()
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, captions):
        for caption in captions:
            tokens = split_sentence(caption)
            self.word_counts.update(tokens)

        # Adding PAD to tokenizer list
        self.word_index["PAD"] = 0
        self.index_word[0] = "PAD"
        self.word_index[self.oov_token] = 1
        self.index_word[1] = self.oov_token

        most_common = self.word_counts.most_common(
            self.num_words - 2 if self.num_words else None
        )

        idx = 2
        for word, _ in most_common:
            if word not in self.word_index:
                self.word_index[word] = idx
                self.index_word[idx] = word
                idx += 1

    def texts_to_sequences(self, captions):
        sequences = []
        unk_idx = self.word_index[self.oov_token]

        for caption in captions:
            tokens = split_sentence(caption)
            seq = [self.word_index.get(token, unk_idx) for token in tokens]
            sequences.append(seq)

        return sequences


def pad_sequences(sequences, maxlen, padding="post"):
    arr = np.zeros((len(sequences), maxlen), dtype=np.int64)

    for i, seq in enumerate(sequences):
        seq = seq[:maxlen]
        if padding == "post":
            arr[i, : len(seq)] = seq
        else:
            arr[i, -len(seq) :] = seq

    return arr


class FeatureCaptionDataset(Dataset):
    def __init__(self, df, caption_vector, feature_dict):
        self.df = df.reset_index(drop=True)
        self.caption_vector = torch.tensor(caption_vector, dtype=torch.long)
        self.feature_dict = feature_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]["Image_ID"]
        features = torch.tensor(self.feature_dict[image_id], dtype=torch.float32)
        caption = self.caption_vector[idx]
        return features, caption, image_id
