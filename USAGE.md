# Usage Guide (Setup + Preprocessing)

## 0. Clone Repo

```bash
git clone <repo-url>
cd <repo-name>
```

---

## 1. Create Environment

```bash
conda env create -f environment.yml
conda activate show-attend-tell
```

If conda fails:

```bash
pip install torch torchvision numpy pandas scikit-learn pillow tqdm
```

---

## 2. Dataset Setup

Download Flickr8k and place it here:

```
data/flickr8k/
```

The folder should follow this structure:

```
data/flickr8k/
├── Images/
├── Flickr8k.token.txt
├── Flickr_8k.trainImages.txt
├── Flickr_8k.devImages.txt
├── Flickr_8k.testImages.txt
```

Notes:

* `Images/` should contain all images (jpg)

---

## 3. Run Preprocessing

From the project root:

```bash
python src/preprocess_flickr8k.py
```

---

## 4. Output

After running, this folder will be created:

```
data/flickr8k/processed/
```

Files generated:

* tokenizer + metadata
* caption vectors (train/val/test)
* VGG16 features
* ResNet50 features

---

## 5. Common Issues

### FileNotFoundError

* Make sure dataset is in `data/flickr8k/`
* Make sure you are running from project root

---

### CUDA out of memory

Edit in `preprocess_flickr8k.py`:

```python
BATCH_SIZE = 32
```

---

### Slow runtime

* First run takes ~10–30 minutes (feature extraction)
* This is expected

---

## 6. Quick Check

To confirm that the image and captions paths exist before running:

```python
import os
print(os.path.exists("data/flickr8k/Images"))
print(os.path.exists("data/flickr8k/Flickr8k.token.txt"))
```

Both should return `True`
