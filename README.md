# Show, Attend, Tell

Cheryl Lam, Yanwei Liu

# Introduction

This repository contains a reimplementation of [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044) (Xu et al., 2015). 

Prior work approached image captioning using an encoder-decoder framework, where a CNN encodes the image into a vectorial representation that is then passed to a RNN to generate the caption. However, these approaches represented the image as a **single static feature vector** from the convnet, compressing away the spatial information that could be useful for richer, more descriptive captions. **Show, Attend, Tell** introduced attention into the encoder-decoder framework, allowing the decoder to dynamically focus on differenet spatial regions of the image as it generates each word; that is, the model learns *where to look* at each timestep. The paper presents two variants: **soft attention**, trained end-to-end via backpropagation; and **hard attention**, trained with the REINFORCE algorithm. In addition to improving performance over prior captioning models, the attention mechanism provides **interpretability**, as the learned attention weights can be visualized to show which parts of the image the model attends to while generating each word.

---

## Chosen Result

We reproduce **Table 1** from the original paper: BLEU-1 through BLEU-4 and METEOR scores for soft and hard attention models trained on Flickr8k. The paper's reported scores for Flickr8k are:

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
|---|---|---|---|---|---|
| Soft Attention | 67 | 44.8 | 29.9 | 19.5 | 18.93 |
| Hard Attention | 67 | 45.7 | 31.4 | 21.3 | 20.30 |

These metrics are the primary evidence that attention-based captioning outperforms prior methods across all metrics. We additionally compare VGG-19 vs. ResNet-50 as feature extractors to evaluate the impact of encoder representation on captioning quality.



---

## Repository Structure

```
show-attend-tell/
├── code/                  
│   ├── preprocess.py      # Data preprocessing & feature extraction
│   ├── project.py         # Python notebook to train
│   ├── models.py          # Encoder, Attention, Decoder modules
│   ├── train.py           # Training loop (soft & hard)
│   └── visualize.py       # Attention heatmap visualization
│
├── data/
│   └── flickr8k/          # Dataset (see setup instructions)
│
├── results/               # Generated outputs & visualizations
├── checkpoints/           # Saved model weights
├── poster/                # Poster
├── report/                # 2-page report
│
├── requirements.txt       
├── README.md              
└── .gitignore
```

---

## Reimplementation Details

**Architecture.** We implement the encoder-decoder framework from the paper. The encoder is a pretrained CNN producing annotation vectors `a = {a1, ..., aL}`, one per spatial location. The decoder is an LSTM that generates one word per timestep conditioned on a context vector `zˆt` derived from attention over the encoder features, the previous hidden state, and the previous word.

**Encoder.** We support two feature extractors, both pretrained on ImageNet and frozen during training:
- **VGG-19** (original paper): features extracted before the final pooling layer → 14×14 spatial map, `L=196`, `D=512`
- **ResNet-50** (our addition): features from the final convolutional block → 7×7 spatial map, `L=49`, `D=2048`

**Soft Attention.** The context vector is a weighted sum over spatial features. It is fully differentiable and trained end-to-end via backpropagation. We include the doubly stochastic regularization term from the paper to encourage the model to attend to all image regions over the course of generation as it was shown to improve BLEU. A learned gating scalar further modulates the context vector.

**Hard Attention.** Attention location is treated as a latent variable, and hard attention is trained by maximizing a variational lower bound on the marginal log-likelihood (Equations 10–12 in the paper), equivalent to REINFORCE. Variance is reduced via a moving average baseline, an entropy regularization term on the attention distribution, and with probability 0.5 replacing the sampled location with its expected value.

**Decoder.** Output is computed via a deep output layer (Equation 7 in the paper).

**Training.** The original paper used RMSProp for Flickr8k. We found Adam performed comparably. Early stopping is based on validation BLEU-4, consistent with the paper. Beam search with beam size 7 is used at inference.

**Dataset.** We use Flickr8k, which consists of 8,000 images with 5 human-annotated captions each, using the standard train/val/test splits.

---

## Reproduction Steps

### 1. Requirements

```bash
pip install -r requirements.txt
```
### 2. Dataset

Download the Flickr8k dataset and place it under `data/flickr8k/` with the following structure:

```
data/flickr8k/
├── Images/
├── Flickr8k.token.txt
├── Flickr_8k.trainImages.txt
├── Flickr_8k.devImages.txt
└── Flickr_8k.testImages.txt
```

### 3. Preprocessing

Extracts VGG-19 and ResNet-50 features, builds the tokenizer, pads captions, and saves everything to `data/flickr8k/processed/`.

```bash
python code/preprocess.py
```

### 4. Training

We trained on Google Colab using a T4 GPU, which took approximately 5 hours to complete.

Train a soft attention model with VGG-19 features:

```bash
python code/train.py --feature_extractor vgg
```

Train a hard attention model:

```bash
python code/train.py --feature_extractor vgg --hard_attention
```

Train with ResNet-50 features:

```bash
python code/train.py --feature_extractor resnet
```

Checkpoints are saved to `checkpoints/`.

### 5. Visualization

Generate attention heatmaps for a given image and checkpoint:

```bash
# Soft attention
python code/visualize.py \
  --image_path data/flickr8k/Images/<image>.jpg \
  --checkpoint checkpoints/soft_vgg_5.pth \
  --feature_extractor vgg

# Hard attention
python code/visualize.py \
  --image_path data/flickr8k/Images/<image>.jpg \
  --checkpoint checkpoints/hard_vgg_3.pth \
  --feature_extractor vgg \
  --hard_attention
```

Outputs are saved to `results/`.

---

## Results

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
|---|---|---|---|---|---|
| Paper (Soft, VGG-19) | — | — | — | — | — |
| Paper (Hard, VGG-19) | — | — | — | — | — |
| Ours (Soft, VGG-19) | — | — | — | — | — |
| Ours (Hard, VGG-19) | — | — | — | — | — |
| Ours (Soft, ResNet-50) | — | — | — | — | — |
| Ours (Hard, ResNet-50) | — | — | — | — | — |


**Key findings:**
- ResNet-50 outperformed VGG-19 across all metrics.
- Hard attention was substantially more difficult to train than soft attention, requiring more epochs and careful hyperparameter tuning due to its stochastic nature.


**Attention visualizations:**

An important contribution of the paper is that the attention mechanism provides **interpretability**. At each timestep, the attention weights define a distribution over spatial features, which we convert into heatmaps to highlight the regions of the image the model is focusing on when generating each word. These visualizations can be found in `results/outputs`.

![Soft and hard attention heatmaps](results/outputs/54501196_a9ac9d66f2/soft_attn.png)

---

## Conclusion

We successfully reimplemented both soft and hard attention variants from Show, Attend, Tell. Soft attention was straightforward to train end-to-end; hard attention was more difficult to train due to its stochastic nature. ResNet-50 features consistently outperformed VGG-19, demonstrating that encoder quality has a strong impact on captioning performance.

---

## References

- K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R. Zemel, and Y. Bengio. "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention." *ICML*, 2015. [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)
- M. Hodosh, P. Young, and J. Hockenmaier. Flickr8k Dataset.

---

## Acknowledgements

This project was completed as a final project for CS 4782 at Cornell University. We thank Prof. Weinberger, Prof. Ma, and the CS 4782 course staff for their guidance throughout the semester.
