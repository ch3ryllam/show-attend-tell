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


# Get splits
def load_split(filename):
    with open(filename, "r") as f:
        images = f.read().splitlines()
    return images


# Define functions to build vocabulary
def split_sentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace("<start>", " <start> ")
    sentence = sentence.replace("<end>", " <end> ")
    tokens = sentence.split()
    return tokens


def generate_vocabulary(captions):
    words = []

    for sentence in captions:
        sent_words = split_sentence(sentence)
        for word in sent_words:
            words.append(word)
    return sorted(words)


# Shuffle training data
def data_limiter(captions, img_vector):
    img_captions, img_name_vector = shuffle(captions, img_vector, random_state=42)
    return img_captions.reset_index(drop=True), img_name_vector.reset_index(drop=True)


# Create tokenizer for the top words
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


# Create word-to-index and index-to-word mappings.
def print_word_to_index(tokenizer, word):
    print(f"Word = {word}, index = {tokenizer.word_index[word]}")


def print_index_to_word(tokenizer, index):
    print(f"Index = {index}, Word = {tokenizer.index_word[index]}")


def extract_features(image_paths, extractor):
    features = {}
    for i, path in enumerate(image_paths):
        image_id = os.path.basename(path)
        features[image_id] = extractor(path)
        if i % 500 == 0:
            print(f"Processed {i}/{len(image_paths)} images")
    return features


# Dataframes consist of (image, caption) pairings but the same image appears 5 times for each of its captions
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


def main():

    print("Starting Flickr8k preprocessing...")

    # Get Flickr8k paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "flickr8k")

    RAW_IMG_PATH = os.path.join(DATA_DIR, "Images")
    TRAIN_IMG_PATH = os.path.join(DATA_DIR, "Flickr_8k.trainImages.txt")
    VAL_IMG_PATH = os.path.join(DATA_DIR, "Flickr_8k.devImages.txt")
    TEST_IMG_PATH = os.path.join(DATA_DIR, "Flickr_8k.testImages.txt")
    RAW_CAPTION_PATH = os.path.join(DATA_DIR, "Flickr8k.token.txt")
    SAVE_DIR = os.path.join(DATA_DIR, "processed")

    # Reading captions file
    file = open(RAW_CAPTION_PATH, "rb")
    captions_txt = file.read().decode("utf-8")
    file.close()
    img_cap_corpus = captions_txt.split("\n")

    # Create a dataframe which summarizes the image, path & captions as a dataframe
    datatxt = []
    for line in img_cap_corpus:
        col = line.split("\t")  # Seperates columns image and caption with tab

        if len(col) != 2:
            continue

        img_name = col[0].split("#")[0].strip()  # remove #0, #1,...
        caption = col[1].lower().strip()

        # Full image path
        img_path = os.path.join(RAW_IMG_PATH, img_name)

        datatxt.append([img_name, img_path, caption])

    df = pd.DataFrame(datatxt, columns=["Image_ID", "Path", "Caption"])

    uni_filenames = np.unique(df.Image_ID.values)
    print(f"Loaded captions for {len(uni_filenames)} unique images")
    print("Caption distribution (captions per image):")
    print(Counter(Counter(df.Image_ID.values).values()))

    # Get splits
    train_imgs = load_split(TRAIN_IMG_PATH)
    val_imgs = load_split(VAL_IMG_PATH)
    test_imgs = load_split(TEST_IMG_PATH)

    print(
        f"Split sizes (images): train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}"
    )

    # Save splits in dataframes
    train_df = df[df["Image_ID"].isin(train_imgs)].reset_index(drop=True)
    val_df = df[df["Image_ID"].isin(val_imgs)].reset_index(drop=True)
    test_df = df[df["Image_ID"].isin(test_imgs)].reset_index(drop=True)

    print(
        f"Split sizes (captions): train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    # Add the <start> & <end> token to all the captions
    train_df["Caption"] = train_df.Caption.apply(lambda x: f"<start> {x} <end>")
    val_df["Caption"] = val_df.Caption.apply(lambda x: f"<start> {x} <end>")
    test_df["Caption"] = test_df.Caption.apply(lambda x: f"<start> {x} <end>")

    # Store captions
    train_annotations = train_df.Caption
    val_annotations = val_df.Caption
    test_annotations = test_df.Caption

    # Store image paths
    train_img_vector = train_df.Path
    val_img_vector = val_df.Path
    test_img_vector = test_df.Path

    # Create vocabulary including all words in the TRAINING captions
    vocab = generate_vocabulary(train_df.Caption)
    vocabulary = Counter(vocab)

    df_word = pd.DataFrame(list(vocabulary.items()), columns=["word", "count"])
    df_word = df_word.sort_values(by="count", ascending=False).reset_index(drop=True)

    # Find max length of caption sequence
    max_length = max(train_df.Caption.apply(lambda x: len(x.split())))

    print(f"Max caption length: {max_length}")

    train_img_captions = train_annotations.reset_index(drop=True)
    train_img_vector = train_img_vector.reset_index(drop=True)

    val_img_captions = val_annotations.reset_index(drop=True)
    test_img_captions = test_annotations.reset_index(drop=True)

    top_freq_words = 10000  # paper vocab size is 10,000

    tokenizer = Tokenizer(num_words=top_freq_words)
    tokenizer.fit_on_texts(train_img_captions)

    # Pad each vector to the max_length of the captions and store it to a variable
    train_cap_seqs = tokenizer.texts_to_sequences(train_img_captions)
    val_cap_seqs = tokenizer.texts_to_sequences(val_img_captions)
    test_cap_seqs = tokenizer.texts_to_sequences(test_img_captions)

    train_cap_vector = pad_sequences(train_cap_seqs, maxlen=max_length, padding="post")
    val_cap_vector = pad_sequences(val_cap_seqs, maxlen=max_length, padding="post")
    test_cap_vector = pad_sequences(test_cap_seqs, maxlen=max_length, padding="post")

    print(f"Train caption vector shape: {train_cap_vector.shape}")
    print("Sample caption vectors (first 5):")
    print(train_cap_vector[:5])

    # Shared image transform
    imagenet_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create VGG16 model
    # 14x14x512 feature map of the fourth convolutional layer before max pooling
    vgg_base = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    vgg_model = torch.nn.Sequential(*list(vgg_base.features.children())[:30]).to(
        device
    )  # stop at correct vgg16 layer
    vgg_model.eval()

    print("Loaded VGG16 model")

    # Extracts VGG16 features
    def extract_vgg16_features(img_path):
        img = Image.open(img_path).convert("RGB")
        img = imagenet_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = vgg_model(img)  # dims (1, 512, 14, 14)

        features = features.squeeze(0).permute(1, 2, 0).reshape(-1, features.shape[1])
        features = features.cpu().numpy()

        return features

    # Get unique image paths
    train_img_vector_uniq = train_df.Path.drop_duplicates().tolist()
    val_img_vector_uniq = val_df.Path.drop_duplicates().tolist()
    test_img_vector_uniq = test_df.Path.drop_duplicates().tolist()

    # Extract VGG16 features from Flickr8k images
    print("Extracting VGG16 features...")

    train_vgg = extract_features(train_img_vector_uniq, extract_vgg16_features)
    val_vgg = extract_features(val_img_vector_uniq, extract_vgg16_features)
    test_vgg = extract_features(test_img_vector_uniq, extract_vgg16_features)

    print(
        "Sample VGG feature shape (should be 196 x 512):",
        next(iter(train_vgg.values())).shape,
    )

    # Create ResNet50 model (for comparison to VGG16)
    resnet_base = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    resnet_model = torch.nn.Sequential(*list(resnet_base.children())[:-2]).to(
        device
    )  # stop at earlier layer to get spatial feature vectors like with vgg
    resnet_model.eval()

    print("Loaded ResNet50 model")

    # Extract ResNet50 features
    print("Extracting ResNet50 features...")

    def extract_resnet50_features(img_path):
        img = Image.open(img_path).convert("RGB")
        img = imagenet_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = resnet_model(img)  # dims (1, 2048, 7, 7)

        features = features.squeeze(0).permute(1, 2, 0).reshape(-1, features.shape[1])
        features = features.cpu().numpy()

        return features

    # Extract ResNet50 features from Flickr8k images
    train_resnet = extract_features(train_img_vector_uniq, extract_resnet50_features)
    val_resnet = extract_features(val_img_vector_uniq, extract_resnet50_features)
    test_resnet = extract_features(test_img_vector_uniq, extract_resnet50_features)

    print(
        "Sample ResNet feature shape (shouldbe 49 x 2048):",
        next(iter(train_resnet.values())).shape,
    )

    # Mini-batches = 64
    BATCH_SIZE = 64

    # Datasets with VGG16 feature extractions (used in paper)
    vgg_train_dataset = FeatureCaptionDataset(train_df, train_cap_vector, train_vgg)
    vgg_val_dataset = FeatureCaptionDataset(val_df, val_cap_vector, val_vgg)
    vgg_test_dataset = FeatureCaptionDataset(test_df, test_cap_vector, test_vgg)

    # VGG16 Dataloaders
    vgg_train_dataloader = DataLoader(
        vgg_train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    vgg_val_dataloader = DataLoader(
        vgg_val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    vgg_test_dataloader = DataLoader(
        vgg_test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    print("Created dataloader with VGG16")

    # Datasets with ResNet50 feature extractions (for comparison)
    resnet_train_dataset = FeatureCaptionDataset(
        train_df, train_cap_vector, train_resnet
    )
    resnet_val_dataset = FeatureCaptionDataset(val_df, val_cap_vector, val_resnet)
    resnet_test_dataset = FeatureCaptionDataset(test_df, test_cap_vector, test_resnet)

    resnet_train_dataloader = DataLoader(
        resnet_train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    resnet_val_dataloader = DataLoader(
        resnet_val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    resnet_test_dataloader = DataLoader(
        resnet_test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    print("Created ResNet dataloaders (for comparison)")

    # Save preprocessed outputs
    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(os.path.join(SAVE_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    metadata = {
        "max_length": max_length,
        "vocab_size": len(tokenizer.word_index),
        "top_freq_words": top_freq_words,
    }
    with open(os.path.join(SAVE_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    np.save(os.path.join(SAVE_DIR, "train_cap_vector.npy"), train_cap_vector)
    np.save(os.path.join(SAVE_DIR, "val_cap_vector.npy"), val_cap_vector)
    np.save(os.path.join(SAVE_DIR, "test_cap_vector.npy"), test_cap_vector)

    train_df.to_csv(os.path.join(SAVE_DIR, "train_df.csv"), index=False)
    val_df.to_csv(os.path.join(SAVE_DIR, "val_df.csv"), index=False)
    test_df.to_csv(os.path.join(SAVE_DIR, "test_df.csv"), index=False)

    with open(os.path.join(SAVE_DIR, "train_vgg.pkl"), "wb") as f:
        pickle.dump(train_vgg, f)
    with open(os.path.join(SAVE_DIR, "val_vgg.pkl"), "wb") as f:
        pickle.dump(val_vgg, f)
    with open(os.path.join(SAVE_DIR, "test_vgg.pkl"), "wb") as f:
        pickle.dump(test_vgg, f)

    with open(os.path.join(SAVE_DIR, "train_resnet.pkl"), "wb") as f:
        pickle.dump(train_resnet, f)
    with open(os.path.join(SAVE_DIR, "val_resnet.pkl"), "wb") as f:
        pickle.dump(val_resnet, f)
    with open(os.path.join(SAVE_DIR, "test_resnet.pkl"), "wb") as f:
        pickle.dump(test_resnet, f)

    print(f"All preprocessing outputs saved to: {SAVE_DIR}")
    print("Preprocessing complete! :) ")


if __name__ == "__main__":
    main()
