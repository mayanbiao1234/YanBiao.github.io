##### YanBiao.github.io
# 👋 Imbalanced Learning
We focus on deep imbalanced learning as well as imbalanced learning in open environments, with recent advances in imbalance metrics.
<br />关注不平衡学习，特别是开放环境下的不平衡问题，以及不平衡度量的最新进展！

## 不平衡学习研究分类
We discuss deep imbalance learning, categorizing existing research into resampling, rebalancing losses (cost-sensitive learning), training strategies, data augmentation, feature migration learning, and other methods
<br />我们讨论深度不平衡学习，将现有的研究分为重采样、重新平衡损失（成本敏感学习）、训练策略、数据增广、特征迁移学习以及其他方法


## 综述 Overview
**[1]** Deep Long-Tailed Learning: A Survey (TPAMI 2021) https://arxiv.org/abs/2110.04596

## 重采样以及成本敏感学习 Resampling and cost-sensitive learning 


## 多阶段的训练策略(解耦训练) Multi-stage training strategy
**[1]** Decoupling (ICLR 2020)

**[2]** BBN (CVPR 2020)

**[3]** KCL (ICLR 2021)

**[4]** MiSLAS (CVPR 2021)

**[5]** Distribution Alignment: A Unified Framework for Long-tail Visual Recognition 分布对齐:长尾视觉识别的统一框架(CVPR2021) https://arxiv.org/abs/2103.16370

**[6]** A novel three-stage training strategy for long-tailed classification 一种新颖的长尾分类三阶段训练策略 https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2104.09830

**[7]** Learning From Long-Tailed Data With Noisy Labels 带噪声标签的长尾数据学习(2021年提交到arxiv) https://arxiv.org/abs/2108.11096

**[8]** Delving into Sample Loss Curve to Embrace Noisy and Imbalanced Data 深入研究样本损失曲线来包含噪声和不平衡数据 (AAAI 2022) https://arxiv.org/abs/2201.00849

## 数据增广 Data Augmentation
**[1]** Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment 通过高斯扰动调整logit的方法(构造“特征云”)，解决长尾分类中各类样本严重不平衡的问题 (CVPR 2022) https://www.techrxiv.org/articles/preprint/Long-tailed_Visual_Recognition_via_Gaussian_Clouded_Logit_Adjustment/17031920/1

**[2]** Improving Calibration for Long-Tailed Recognition 改进的长尾识别校准(CVPR 2021) https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_Improving_Calibration_for_Long-Tailed_Recognition_CVPR_2021_paper.pdf 这篇文章指出使用数据mixup对特征学习是有益的，但是会损害分类器学习，当使用解耦训练策略时，也会对分类器学习有害，但可以忽略不计。[![知乎](https://img.shields.io/badge/知乎-white?logo=zhihu)](https://zhuanlan.zhihu.com/p/419911014)

**[3]** FASA: Feature Augmentation and Sampling Adaptation for Long-Tailed Instance Segmentation 长尾实例分割的特征增强和采样自适应 (ICCV 2021)https://arxiv.org/abs/2102.12867

**[4]** Feature Generation for Long-tail Classification 长尾分类的特征生成 (ICVGIP'21) https://arxiv.org/abs/2111.05956

**[5]** Feature Space Augmentation for Long-Tailed Data 用于长尾数据的特征空间增广 (ECCV2020) https://arxiv.org/abs/2008.03673 这篇文章也可以放在特征迁移学习中

**[6]** Deep Representation Learning on Long-tailed Data: A Learnable Embedding Augmentation Perspective 为尾部类别构造云、用“特征云”来充实尾部类的方法（CVPR 2020）https://arxiv.org/abs/2002.10826 **备注:** 和[1]基本一样，而且这篇论文研究似乎更加深入，[1]有抄袭的现象吗？这篇论文也可以被划分进入特征迁移学习的类别。

**[7]** Remix: Rebalanced Mixup 重新平衡的Mixup （ECCV WorkShop 2020）https://arxiv.org/pdf/2007.03943v3.pdf

## 特征迁移学习 Feature transfer learning
**[1]** Feature Transfer Learning for Face Recognition With Under-Represented Data 针对欠表示数据人脸识别的特征迁移学习(CVPR 2019)https://ieeexplore.ieee.org/document/8953809/citations#citations

**[2]** Deep Representation Learning on Long-tailed Data: A Learnable Embedding Augmentation Perspective 基于长尾数据的深度表示学习:一种可学习的嵌入增强视角(CVPR  2020)https://arxiv.org/abs/2002.10826

**[3]** Feature Space Augmentation for Long-Tailed Data 用于长尾数据的特征空间增广 (ECCV2020) https://arxiv.org/abs/2008.03673 这篇文章也可以放在特征迁移学习中

**[4]** Deep Representation Learning on Long-tailed Data: A Learnable Embedding Augmentation Perspective 为尾部类别构造云、用“特征云”来充实尾部类的方法（CVPR 2020）https://arxiv.org/abs/2002.10826 **备注:** 和[1]基本一样，而且这篇论文研究似乎更加深入，[1]有抄袭的现象吗？这篇论文也可以被划分进入特征迁移学习的类别。

**[5]** RSG: A Simple but Effective Module for Learning Imbalanced Datasets 一个简单但有效的学习不平衡数据集的模块(cvpr2021) https://arxiv.org/abs/2106.09859

**[6]** M2m: Imbalanced Classification via Major-to-minor Translation (CVPR2020):除了特征层面的头到尾的转移，头尾转换(M2m)[79]提出通过基于扰动的优化，将头类样本转换为尾类样本来增加尾类。通过基于扰动的优化，将头类样本翻译成尾类样本，这与对抗性攻击本质上相似。转换后的尾类样本将被用来构建一个更平衡的训练集进行模型训练。https://arxiv.org/abs/2004.00431

**[7]** GistNet: a Geometric Structure Transfer Network for Long-Tailed Recognition  一种用于长尾识别的几何结构迁移网络 (ICCV2021):在分类器层面进行头到尾的转移。通过用头类相对较大的分类器几何信息来增强尾类的分类器权重，GIST能够获得更好的尾部类性能。https://arxiv.org/pdf/2105.00131.pdf

**[8]** Label-Aware Distribution Calibration for Long-tailed Classification 标签感知分布校准用于长尾分类 (似乎未正式发表) https://arxiv.org/abs/2111.04901

**[9]** Class-Distribution-Aware Calibration for Long-Tailed Visual Recognition 长尾视觉识别的类分布感知校准(ICML 2021 WorkShop) https://arxiv.org/abs/2109.05263

**[10]** Sharing Representations for Long Tail Computer Vision Problems 长尾计算机视觉问题的共享表示(ICMI 2015) https://dl.acm.org/doi/10.1145/2818346.2818348

## 集成学习 
**[1]** RIDE

**[2]** Trustworthy Long-Tailed Classification 值得信赖的长尾分类 （CVPR 2022）https://arxiv.org/abs/2111.09030

## logit Adiustment
**[1]** Adaptive Logit Adjustment Loss for Long-Tailed Visual Recognition 长尾视觉识别的自适应Logit调整损失(AAAI2022) https://arxiv.org/pdf/2104.06094.pdf

**[2]** Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment 通过高斯扰动调整logit的方法(构造“特征云”)，解决长尾分类中各类样本严重不平衡的问题 (CVPR 2022) https://www.techrxiv.org/articles/preprint/Long-tailed_Visual_Recognition_via_Gaussian_Clouded_Logit_Adjustment/17031920/1

## 其他方法 Other methods
**[1]** Memory-based Jitter: Improving Visual Recognition on Long-tailed Data with Diversity In Memory 基于内存的抖动：提高对具有内存多样性的长尾数据的视觉识别 (CVPR 2021) https://arxiv.org/abs/2008.09809
<br />备注：这篇论文将尾部类别的历史特征保存下来，以实现和头部类别在数量上的平衡

**[2]** Long-tailed Distribution Adaptation 长尾分布适应 （ACM MM 2021）https://arxiv.org/pdf/2110.02686.pdf

**[3]** Calibrating Class Activation Maps for Long-Tailed Visual Recognition 校准类激活映射的长尾视觉识别 (2021年提交到arxiv) https://arxiv.org/abs/2108.12757

## 标准数据集
**CIFAR10-LT**, **CIFAR100-LT**, **ImageNet-LT**, **iNaturalist**, **Places-LT**

### Get in touch
[![GitHub](https://img.shields.io/badge/GitHub-grey?logo=github)](https://github.com/mayanbiao1234)
[![知乎](https://img.shields.io/badge/知乎-white?logo=zhihu)](https://www.zhihu.com/people/ma-yan-biao-73)


