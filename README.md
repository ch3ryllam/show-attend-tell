# Show, Attend, Tell

Cheryl Lam, Yanwei Liu

# Introduction

This repository contains a reimplementation of [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044).

Previous work such as Show and Tell [1] approached image captioning using an encoder-decoder framework, where a CNN encodes the image into a fixed vector representation that is then passed to a RNN to generate the caption. However, representing the image as a single static feature vector from the convnet creates an information bottleneck.

**Show, Attend and Tell** [2] addresses this limitation by introducing an attention mechanism into the encoder-decoder framework, enabling the decoder to dynamically focus on relevant spatial regions while producing each word. Xu et al. present two variants: **soft attention**, which is fully differentiable and trained via backpropagation; and **hard attention**, which samples discrete spatial locations and is trained using REINFORCE. The paper demonstrates improved performance over non-attention baselines on Flickr8k, Flickr30k, and MS-COCO.

---

## Chosen Result

We reproduce the Flickr8k results reported in Table 1 of the original paper, evaluating both soft and hard attention using BLEU-1 through BLEU-4 and METEOR:

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
|---|---|---|---|---|---|
| Soft Attention | 67 | 44.8 | 29.9 | 19.5 | 18.93 |
| Hard Attention | 67 | 45.7 | 31.4 | 21.3 | 20.30 |

These metrics serve as the primary quantitative evidence that attention-based captioning improves over prior methods. Beyond reproducing the original VGG-19-based architecture, we additionally replace the encoder with ResNet-50 to evaluate the impact of a stronger visual feature extractor.


---

## Repository Structure

- `code/`
  - `models.py` - Encoder, attention, and decoder modules
  - `preprocess.py` - Data preprocessing and feature extraction
  - `quick_run.ipynb`- Notebook used for training
  - `train.py` - Training loop for soft/hard attention
  - `visualize.py` - Attention heatmap visualizations

- `data/flickr8k/` - Flickr8k dataset (see setup instructions)
- `results/outputs/`- Generated visualizations
- `poster/` - poster
- `report/` - 2-page report

---

## Reimplementation Details

**Architecture.** A pretrained CNN encoder extracts spatial visual features, while an LSTM decoder generates captions one word at a time conditioned on the previous hidden state, previous word, and an attention-derived context vector $\hat{z}_t.$

**Encoder.** We evaluate two feature extractors, both pretrained on ImageNet and frozen during training:
- **VGG-19** (original paper)
- **ResNet-50** (our extension)

**Soft Attention.** The context vector is computed as a weighted sum over spatial features. This variant is fully differentiable and trained end-to-end via backpropagation. We additionally include the doubly stochastic regularization term from the paper to encourage attention over all image regions during generation.

**Hard Attention.** At each timestep, the model samples a single spatial location instead of attending over all regions. Because sampling is non-differentiable, the model is trained using REINFORCE with a moving-average baseline and entropy regularization for variance reduction and exploration.

**Training.** We use teacher forcing, RMSProp, dropout ($p=0.5$), beam search decoding (beam size $7$), and early stopping on BLEU-4.

**Dataset.** We evaluate on Flickr8k, which contains $8{,}000$ images paired with $5$ human-annotated captions each using the standard train/validation/test splits.


---

## Reproduction Steps

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. Dataset

Download the Flickr8k dataset from Kaggle:  
<https://www.kaggle.com/datasets/adityajn105/flickr8k?select=Images> [3].

Place it under `data/flickr8k/` with the following structure:

```text
data/flickr8k/
├── Images/
├── Flickr8k.token.txt
├── Flickr_8k.trainImages.txt
├── Flickr_8k.devImages.txt
└── Flickr_8k.testImages.txt
```

### 3. Run the Notebook

Open:

```text
code/quick_run.ipynb
```

Run each cell sequentially. The notebook:
- preprocesses Flickr8k,
- extracts VGG-19 and ResNet-50 features,
- trains soft and hard attention models, and
- saves checkpoints to `checkpoints/`.

Training was performed on Google Colab using a T4 GPU. Each model took approximately **3 hours** to train. 
 

### 4. Visualization

Generate attention heatmaps for a given image and checkpoint:

```bash
# Soft attention
python code/visualize.py \
  --image_path data/flickr8k/Images/<image>.jpg \
  --checkpoint checkpoints/soft_resnet.pth \
  --feature_extractor resnet

# Hard attention
python code/visualize.py \
  --image_path data/flickr8k/Images/<image>.jpg \
  --checkpoint checkpoints/hard_resnet.pth \
  --feature_extractor resnet \
  --hard_attention
```

Outputs are saved to `results/`.

---

## Results

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
|---|---|---|---|---|---|
| Paper (Soft, VGG-19) | 67 | 44.8 | 29.9 | 19.5 | 18.93 |
| Paper (Hard, VGG-19) | 67 | 45.7 | 31.4 | 21.3 | 20.30 |
| Ours (Soft, VGG-19) | 65.67 | 45.67 | 32.14 | 22.61 | 20.72 |
| Ours (Hard, VGG-19) | 64.72 | 43.57 | 29.87 | 20.46 | 19.28 |
| **Ours (Soft, ResNet-50)** | **68.01** | **48.30** | **34.39** | **24.20** | **21.66** |
| **Ours (Hard, ResNet-50)** | **68.31** | **48.75** | **34.67** | **24.40** | **22.04** |


**Findings:**
- Our soft attention VGG implementation exceeds the paper’s reported BLEU-4 and METEOR scores, while hard attention VGG falls slightly below the paper's results.
- Hard attention was substantially harder to train because stochastic attention sampling and REINFORCE introduce high variance, making optimization more sensitive to hyperparameters.
- Replacing VGG-19 with ResNet-50 improved both soft and hard attention performance across all metrics. ResNet-50 improved BLEU-4 by approximately **1.6** for soft attention and **3.9** for hard attention over our VGG baselines, likely because the deeper residual encoder produces more discriminative semantic features.
- With a stronger encoder, hard attention nearly matched soft attention in BLEU-4 and slightly exceeded it in METEOR, suggesting that the performance gap between the two attention variants narrows when visual features improve.

**Attention visualizations:**

An important contribution of the paper is that the attention mechanism provides **interpretability**. At each timestep, the attention weights define a distribution over spatial features, which we convert into heatmaps to highlight the regions of the image the model is focusing on when generating each word. These visualizations can be found in `results/outputs`.

![Soft and hard attention heatmaps](results/outputs/54501196_a9ac9d66f2_poster/soft_attn.png)

---

## Conclusion

We successfully reimplemented both soft and hard attention variants from Show, Attend, Tell. Soft attention was straightforward to train end-to-end; hard attention was more difficult to train due to its stochastic nature. ResNet-50 features consistently outperformed VGG-19, demonstrating that encoder quality has a strong impact on captioning performance.

---


## References

[1] O. Vinyals, A. Toshev, S. Bengio, and D. Erhan, *“Show and Tell: A Neural Image Caption Generator,”* CoRR, vol. abs/1411.4555, 2014. [Online]. Available: http://arxiv.org/abs/1411.4555

[2] K. Xu, J. Ba, R. Kiros, K. Cho, A. C. Courville, R. Salakhutdinov, R. S. Zemel, and Y. Bengio, *“Show, Attend and Tell: Neural Image Caption Generation with Visual Attention,”* CoRR, vol. abs/1502.03044, 2015. [Online]. Available: http://arxiv.org/abs/1502.03044

[3] M. Hodosh, P. Young, and J. Hockenmaier, *“Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics,”* Journal of Artificial Intelligence Research, vol. 47, pp. 853–899, 2013.

[4] A. Lavie and A. Agarwal, *“METEOR: An Automatic Metric for MT Evaluation with High Levels of Correlation with Human Judgments,”* in *Proceedings of the Second Workshop on Statistical Machine Translation*, Prague, Czech Republic, 2007, pp. 228–231. [Online]. Available: https://aclanthology.org/W07-0734/

[5] K. He, X. Zhang, S. Ren, and J. Sun, *“Deep Residual Learning for Image Recognition,”* in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[6] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, *“ImageNet: A Large-Scale Hierarchical Image Database,”* in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2009, pp. 248–255.

---

## Acknowledgements

This project was completed as a final project for CS 4782 at Cornell University. We thank Prof. Weinberger, Prof. Ma, and the CS 4782 course staff for their guidance throughout the semester.

[OpenAI](https://chatgpt.com/) and [Anthropic](https://claude.ai/) were used for debugging assistance.
