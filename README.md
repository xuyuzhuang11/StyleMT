# Machine Translation with Style Customization

[![GitHub Code License](https://img.shields.io/github/license/xuyuzhuang11/StyleMT)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/xuyuzhuang11/StyleMT)](https://github.com/xuyuzhuang11/StyleMT/commits/main)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/xuyuzhuang11/StyleMT/pulls)
[![GitHub release version](https://img.shields.io/github/v/release/xuyuzhuang11/StyleMT)](https://github.com/xuyuzhuang11/StyleMT)

This repository is the code and data for the paper "[Pluggable Neural Machine Translation Models via Memory-Augmented Adapters](https://arxiv.org/abs/2307.06029)" accepted by LREC-COLING 2024. The approach proposed in this paper is for customizing the style or domain property in machine translations. The main idea is adding pluggable and memory-augmented adapters (ours) into pretrained neural machine translation model and training the adapters with the proposed consistent training loss. In particular, the memory is constructed using multi-granularity phrases and the phrases is divided into different layers of Transformer model.

## MTSC Data

Machine Translation with Style Customization (MTSC) is a specialized dataset used for training and evaluating stylized machine translation models. This dataset includes stylized tasks in two different languages: Lu Xun style (Chinese, Zh) and Shakespeare style (English, En). Each style comes with a training set, a validation set, and a test set. For the training and validation sets, they consist of monolingual data in the target style. For the test set, it consists of data in the target style and manually annotated translations in the source language. The statistics for each style are as follows:

| Style | Src-side Language | Tgt-side Language | Training Set | Validation Set | Test Set |
| ----- | ----------------- | ----------------- | ------------ | -------------- | ------- |
| Lu Xun | En | Zh | 37K | 500 | 500 |
| Shakespeare | Zh | En | 20K | 500 | 500 |



## Installation

## Training

## Testing

## Issues

## License

## Citation

If you found this work useful, either the proposed method or our dataset MTSC, please consider citing:

```bibtex
@inproceedings{xu2023pluggable,
  title={Pluggable Neural Machine Translation Models via Memory-augmented Adapters},
  author={Xu, Yuzhuang and Wang, Shuo and Li, Peng and Liu, Xuebo and Wang, Xiaolong and Liu, Weidong and Liu, Yang},
  booktitle={Proceedings of the 2024 joint international conference on computational linguistics, language resources and evaluation},
  year={2023}
}
```

## Note

This repository is in preparing. Please wait for 3 days.