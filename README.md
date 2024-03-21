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

We implement our model based on the open-source implementation of the transformer, [fairseq](https://github.com/facebookresearch/fairseq). To use these codes, you should clone this repository to your local machine, enter the code folder and install it.

```bash
git clone https://github.com/xuyuzhuang11/StyleMT.git
cd StyleMT/fairseq-pro-StyleMT
pip install -e .

```

Then, you can perform training or testing using these resources. Note that you must have modified the arguments in each script to suit your environment.

## Training

You should firstly train the base NMT model using the script in `scripts/En-Zh` or `scripts/Zh-En`.

Then, you can build the datastores of different layers:

```bash
bash prepare_KVs_of_layers.sh

```

Finally, you can train the memory-augmented adapters. Build the checkpoint and train it like this:

```bash
bash prepare_ckpt.sh
bash tune-t2.sh

```

## Testing

You can use the following script to evaluate the translation quality:

```bash
bash test-t2.sh

```

The best scores of our proposed style customization with MTSC dataset are as follows till now:

| Style | BLEU | PPL | Classifier Score |
| ----- | ---- | --- | ---------------- |
| Lu Xun | 21.3 | 199.6 | 53.2 |
| Shakespeare | 21.8 | 95.1 | 85.2 |

## Issues

Despite the considerable amount of time we spent on creating the MTSC dataset, low-level errors are still difficult to avoid. We welcome other researchers to help us improve the overall quality of the dataset, including correcting flaws, adding new languages, introducing new styles, etc. You are invited to contribute your insights through issues or pull requests (PRs). Furthermore, if you find any flaws in the code, you are welcome to get in touch with us.

## License

This code and dataset are released under the MIT license, allowing you to freely use them within its framework. However, please note not to forget to cite the work as follows.

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
