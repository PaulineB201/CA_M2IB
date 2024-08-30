# Multi-Modal Information Bottleneck Attribution with Cross-Attention Guidance

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