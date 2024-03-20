# Machine Translation with Style Customization

[![GitHub Code License](https://img.shields.io/github/license/xuyuzhuang11/StyleMT)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/xuyuzhuang11/StyleMT)](https://github.com/xuyuzhuang11/StyleMT/commits/main)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/xuyuzhuang11/StyleMT/pulls)
[![GitHub release version](https://img.shields.io/github/v/release/xuyuzhuang11/StyleMT)](https://github.com/xuyuzhuang11/StyleMT)

This repository is the code and data for the paper "[Pluggable Neural Machine Translation Models via Memory-Augmented Adapters](https://arxiv.org/abs/2307.06029)" accepted by LREC-COLING 2024. The approach proposed in this paper is for customizing the style or domain property in machine translations. The main idea is adding pluggable and memory-augmented adapters (ours) into pretrained neural machine translation model and training the adapters with the proposed consistent training loss. In particular, the memory is constructed using multi-granularity phrases and the phrases is divided into different layers of Transformer model.

# This repo is set up on 20th March.