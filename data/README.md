---
license: mit
---

# Datasets: 

Base-to-Novel: [ImageNet-1K](https://image-net.org/challenges/LSVRC/2012/index.php), [Caltech101](https://data.caltech.edu/records/mzrjq-6wc02), [Oxford Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/), [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [Food101](https://vision.ee.ethz.ch/datasets_extra/food-101/), [FGVC Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [SUN397](http://vision.princeton.edu/projects/2010/SUN/), [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [EuroSAT](https://github.com/phelber/EuroSAT), [UCF101](https://www.crcv.ucf.edu/data/UCF101.php).

Domain Generalization: [ImageNet-V2](https://github.com/modestyachts/ImageNetV2), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch), [ImageNet-Adversarial](https://github.com/hendrycks/natural-adv-examples), [ImageNet-Rendition](https://github.com/hendrycks/imagenet-r).

Due to various factors, the links to some datasets may be outdated or invalid.

To make it easy for you to download these datasets, we maintain a repository on HuggingFace, which contains all the datasets to be used (except ImageNet). Each dataset also includes the corresponding split_zhou_xx.json file.

# Instructions for How to download these datasets:

## Using the huggingface-cli command-line tool:

Install the CLI tool if not already installed.

`pip install -U huggingface-hub`

Download the datasets.

`huggingface-cli download zhengli97/prompt_learning_dataset`

<hr/>

# Some projects from our lab may familiarize you with prompt learning:

- Open Source Paper List: https://github.com/zhengli97/Awesome-Prompt-Adapter-Learning-for-VLMs
- 中文视频解读：《视觉语言模型CLIP的提示学习方法研究》，[链接](https://www.techbeat.net/talk-info?id=915)
- Published Papers:
  - **Advancing Textual Prompt Learning with Anchored Attributes.** ICCV 2025. [[Paper](https://arxiv.org/abs/2412.09442)] [[Project Page](https://zhengli97.github.io/ATPrompt/)] [[Code](https://github.com/zhengli97/ATPrompt)] [[中文解读](https://zhuanlan.zhihu.com/p/11787739769)] [[中文翻译](https://github.com/zhengli97/ATPrompt/blob/main/docs/ATPrompt_chinese_version.pdf)]
  - **PromptKD: Unsupervised Prompt Distillation for Vision-Language Models.** CVPR 2024. [[Paper](https://arxiv.org/abs/2403.02781)] [[Project Page](https://zhengli97.github.io/PromptKD)] [[Code](https://github.com/zhengli97/PromptKD)] [[中文解读](https://zhuanlan.zhihu.com/p/684269963)] [[中文翻译](https://github.com/zhengli97/PromptKD/blob/main/docs/PromptKD_chinese_version.pdf)]
  - **Cascade Prompt Learning for Vision-Language Model Ddaptation.** ECCV 2024. [[Paper](https://arxiv.org/abs/2409.17805)] [[Code](https://github.com/megvii-research/CasPL)] [[中文解读](https://zhuanlan.zhihu.com/p/867291664)]
  - **Fine-Grained Visual Prompting.** NeurIPS 2023. [[Paper](https://arxiv.org/abs/2306.04356)] [[Code](https://github.com/ylingfeng/FGVP)]




