# Multi-Modal Information Bottleneck Attribution with Cross-Attention Guidance

**Abstract**
For the progression of interpretable machine learning, particularly in the intersection of vision and language, ensuring transparency and comprehensibility in model decisions is crucial. This work introduces an enhancement to the Multi-modal Information Bottleneck attribution method by integrating cross-attention mechanisms. This targets the core challenge of improving the interpretability of vision-language pretrained models, such as CLIP, by fostering more discerning and relevant latent representations. The proposed method filters and retains essential information across modalities, leveraging cross-attention to dynamically focus on pertinent visual and textual features for any given context.

![ca-m2ib(3)](https://github.com/user-attachments/assets/445fc20a-c7ba-428d-939e-a6ca92eb50a6)


### Installation 

Clone repository and install necessary packages, run: 

```
python install -r requirements.txt

```

### Dataset 

This work uses the [Conceptual Caption](https://ai.google.com/research/ConceptualCaptions/), a large dataset of ~3.3M images annotated with captions and [MS-CXR](https://physionet.org/content/ms-cxr/0.1/), a dataset of 1162 image–sentence pairs of bounding boxes and corresponding captions, across eight different cardiopulmonary radiological findings. 

### Usage 

We provide model and example, run: 

```
python run.py
```

### Acknowledgments 

This work is based on [M2IB](https://github.com/YingWANGG/M2IB/tree/main), [IBA](https://github.com/bazingagin/IBA) and [IBA]( https://github.com/BioroboticsLab/IBA). We thank the authors for their contribution and releasing their code.
