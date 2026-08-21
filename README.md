# Progressive Hybrid Pseudo-Labeling for Unsupervised Domain Adaptation with Ascending Low-Rank Adaptation



------

## Highlights

![Architecture](https://github.com/el2k/PHPL/blob/main/Architecture.png)
> Abstract: Unsupervised domain adaptation (UDA) based on large vision-language models (VLMs) has recently demonstrated strong generalization ability, yet it remains fundamentally challenged by noisy pseudo-labels and inefficient adaptation under large domain shifts. In this paper, we propose Progressive Hybrid Pseudo-Labeling for UDA with Ascending Low-Rank Adaptation (termed as PHPL), a parameter-efficient paradigm that addresses these challenges from two complementary perspectives. 
(1) We introduce a progressive hybrid pseudo-labeling strategy that constructs target-domain supervision by fusing predictions from a frozen teacher model and an adaptive student model with a progressive weighting scheme. By gradually transferring predictive responsibility from the teacher to the student during training, PHPL effectively mitigates early-stage pseudo-label noise and stabilizes self-training under large domain shifts. 
(2) To enable efficient and stable adaptation of large VLMs, we propose an ascending low-rank adaptation strategy that allocates LoRA capacity in a depth-aware manner. Specifically, larger low-rank updates are assigned to deeper, semantically richer layers, while shallow layers remain lightly parameterized, striking a favorable balance between parameter efficiency and representational expressiveness. 
We conduct extensive experiments on five widely-used UDA benchmarks, including Office-Home, Office-31, VisDA-2017, Mini-DomainNet, and DomainNet. Experimental results verify that PHPL consistently achieves higher performance across various cross-domain scenarios compared with existing CNN, Transformer, and VLMs-based solutions. Notably, PHPL demonstrates strong robustness on highly challenging large-scale conditions while requiring significantly less computational overhead, validating the effectiveness and scalability of the proposed lightweight adaptation paradigm. The code is available at https://github.com/el2k/PHPL.

## Main Contributions

- **New perspective：** We conceptualize the UDA process not merely as model updating, but as a dynamic evolution of two critical resources: reliable pseudo-label supervision and parameter-efficient learning capacity. The reliable supervision and the high parameter budget are progressively directed towards where they are needed most: initially to stabilize learning, and later towards the deeper, semantically critical network layers for refining domain-specific representations, leading to robust and efficient adaptation..
- **Novel method：** We propose two synergistic techniques: (1) a progressive hybrid pseudo-labeling strategy, which generates evolving and target-domain-adaptive supervision by fusing predictions from a stable teacher model and a progressively adapting student model; (2) an ascending LoRA strategy, which allocates increasing parameter capacity to deeper layers for higher-level representations. These strategies work in concert: the dynamic pseudo-labels provide a continuously refined learning target, while the ascending-rank structure enables a cost-efficient, progressive refinement of the model's most semantically critical representations. 
    %We propose PHPL, a progressive hybrid pseudo-labeling framework with ascending low-rank adaptation, which enables reliable target-domain supervision and efficient, scalable adaptation of pretrained vision--language models without introducing heavy architectural modifications.
- **High Performance：** We conduct extensive experiments on five widely-used UDA benchmarks, including 
    Office-Home, Office-31, VisDA-2017, Mini-DomainNet and DomainNet. Experimental results demonstrate that our proposed PHPL consistently outperforms existing state-of-the-art methods under large domain shifts. Meanwhile, PHPL achieves superior robustness with significantly lower computational complexity, validating the effectiveness and scalability of the proposed lightweight adaptation paradigm.

------




## Installation



For installation and other package requirements, please follow the instructions as follows. This codebase is tested on Ubuntu 18.04 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

- Setup conda environment.

```
# Create a conda environment
conda create -y -n phpl python=3.8

# Activate the environment
conda activate phpl

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/get-started/previous-versions/ if your cuda version is different
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```



- Install dassl library.

```
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch

# Install dependencies
pip install -r requirements.txt

```



- Clone PHPL code repository and install requirements.

```
# Clone PHPL code base
git clone https://github.com/el2k/PHPL.git
cd PHPL

# Install requirements
pip install -r requirements.txt
```



## Data preparation



Please follow the instructions as follows to prepare all datasets. Datasets list:

- [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?pli=1&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)
- [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code)
- [VisDA-2017](http://ai.bu.edu/visda-2017/#download)
- [DomainNet](http://ai.bu.edu/M3SDA/)
- [miniDomainNet](https://arxiv.org/abs/2003.07325)
------

## Training and Evaluation

Please follow the instructions for training, evaluating and reproducing the results. Firstly, you need to **modify the directory of data by yourself**.

### Training



```
# Example: trains on Office-Home dataset, and the source domian is art and the target domain is clipart (a-c)
bash scripts/phpl/main_phpl.sh officehome b32_ep10_officehome PHPL ViT-B/16 2 a-c 0
```



### Evaluation



```
# evaluates on Office-Home dataset, and the source domian is art and the target domain is clipart (a-c)
bash scripts/phpl/eval_phpl.sh officehome b32_ep10_officehome PHPL ViT-B/16 2 a-c 0
```



The details are at each method folder in [scripts folder]([PHPL/scripts at main · el2k/PHPL (github.com)](https://github.com/el2k/PHPL/tree/main/scripts)).


## Acknowledgements



Our style of reademe refers to [PDA](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment). And our code is based on [CoOp and CoCoOp](https://github.com/KaiyangZhou/CoOp), [DAPL](https://github.com/LeapLabTHU/DAPrompt/tree/main) , [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning)  , [PDA](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment) , [PMCC](https://github.com/246dxw/PMCC) and [CDU](https://github.com/1d1x1w/CDU) etc. repository. We thank the authors for releasing their code. If you use their model and code, please consider citing these works as well. Supported methods are as follows:

| Method       | Paper                                          | Code                                                         |
| ------------ | ---------------------------------------------- | ------------------------------------------------------------ |
| CoOp         | [IJCV 2022](https://arxiv.org/abs/2109.01134)  | [link](https://github.com/KaiyangZhou/CoOp)                  |
| CoCoOp       | [CVPR 2022](https://arxiv.org/abs/2203.05557)  | [link](https://github.com/KaiyangZhou/CoOp)                  |
| VPT          | [ECCV 2022](https://arxiv.org/abs/2203.17274)  | [link](https://github.com/KMnP/vpt)                          |
| IVLP & MaPLe | [CVPR 2023](https://arxiv.org/abs/2210.03117)  | [link](https://github.com/muzairkhattak/multimodal-prompt-learning) |
| DAPL         | [TNNLS 2023](https://arxiv.org/abs/2202.06687) | [link](https://github.com/LeapLabTHU/DAPrompt)               |
| PDA          | [AAAI 2024](https://arxiv.org/abs/2312.09553)  | [link](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment) |
| PMCC         | [PR 2026](https://www.sciencedirect.com/science/article/abs/pii/S0031320325007551?via%3Dihub)  | [link](https://github.com/246dxw/PMCC)                       |
| CDU          | [PR 2026](https://www.sciencedirect.com/science/article/pii/S0031320325016486?ref=pdf_download&fr=RR-2&rr=9baa13f40b6d2a92)  | [link](https://github.com/1d1x1w/CDU) |
