##### YanBiao.github.io
# 👋 Imbalanced Learning
We focus on deep imbalanced learning as well as imbalanced learning in open environments, with recent advances in imbalance metrics.
<br />关注不平衡学习，特别是开放环境下的不平衡问题，以及不平衡度量的最新进展！

## 不平衡学习研究分类
We discuss deep imbalance learning, categorizing existing research into resampling, rebalancing losses (cost-sensitive learning), training strategies, data augmentation, feature migration learning, and other methods
<br />我们讨论深度不平衡学习，将现有的研究分为重采样、重新平衡损失（成本敏感学习）、训练策略、数据增广、特征迁移学习以及其他方法


## 综述 Overview
**[1]** (**TPAMI 2021**) Deep Long-Tailed Learning: A Survey https://arxiv.org/abs/2110.04596

## 重采样以及成本敏感学习（重新平衡损失） Resampling and cost-sensitive learning 
**[1]**（**CVPR 2022**）Long-Tailed Recognition via Weight Balancing 通过重新平衡进行长尾识别 https://arxiv.org/abs/2203.14197 采用了一个两阶段的训练范式，并提出了一个简单的LTR方法：（1）使用交叉熵损失学习特征 (1)通过调整权重衰减，使用交叉熵损失学习特征，以及(2)通过调整权重衰减和MaxNorm，使用类平衡损失学习分类器。

**[2]**（**CVPR 2022**）Equalized Focal Loss for Dense Long-Tailed Object Detection 用于解决单阶段目标检测长尾问题的均衡版Focal Loss 商汤的论文https://arxiv.org/pdf/2201.02593.pdf] [![知乎](https://img.shields.io/badge/知乎-white?logo=zhihu)](https://zhuanlan.zhihu.com/p/489606679)

**[3]**（**CVPR 2022**）Relieving Long-tailed Instance Segmentation via Pairwise Class Balance 通过两两类平衡缓解长尾实例分割 https://arxiv.org/abs/2201.02784 长尾问题的根源是占比不多的头部类的样本数远多于占比不小的尾部类们。直接在这种数据集上训练的模型，其分类预测会有偏差。易把尾部类样本错分成头部类样本. 现有技术提出一些指标去简单指示偏差, 并进行相应建模，达到某种平衡从而提升效果。要么局限于静态的训练集类别分布，不灵活。要么即使考虑了动态统计量，也只是每个类本身的分类情况，没有考虑到类间错分。PCB 方法使用混淆矩阵维护训练时类间预测偏差信息。对于训练样本，除基本交叉熵损失外，据其类别从混淆矩阵中取得对抗软类标，施以该软类标的交叉熵损失进行纠偏。我们的方法可无缝插入到前沿的长尾实例分割模型中，均取得不俗提升，部分可达领域最佳效果。

## 多阶段的训练策略(解耦训练) Multi-stage training strategy
**[1]** (**ICLR 2020**) Decoupling

**[2]** (**CVPR 2020**) BBN

**[3]** (**ICLR 2021**) KCL

**[4]** (**CVPR 2021**) MiSLAS

**[5]** (**CVPR2021**) Distribution Alignment: A Unified Framework for Long-tail Visual Recognition 分布对齐:长尾视觉识别的统一框架https://arxiv.org/abs/2103.16370

**[6]** A novel three-stage training strategy for long-tailed classification 一种新颖的长尾分类三阶段训练策略 https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2104.09830

**[7]** (2021年提交到arxiv) Learning From Long-Tailed Data With Noisy Labels 带噪声标签的长尾数据学习 https://arxiv.org/abs/2108.11096

**[8]** (**AAAI 2022**) Delving into Sample Loss Curve to Embrace Noisy and Imbalanced Data 深入研究样本损失曲线来包含噪声和不平衡数据https://arxiv.org/abs/2201.00849

## 数据增广 Data Augmentation
**[1]** (**CVPR 2022**) Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment 通过高斯扰动调整logit的方法(构造“特征云”)，解决长尾分类中各类样本严重不平衡的问题 https://www.techrxiv.org/articles/preprint/Long-tailed_Visual_Recognition_via_Gaussian_Clouded_Logit_Adjustment/17031920/1

**[2]** (**CVPR 2021**) Improving Calibration for Long-Tailed Recognition 改进的长尾识别校准https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_Improving_Calibration_for_Long-Tailed_Recognition_CVPR_2021_paper.pdf 这篇文章指出使用数据mixup对特征学习是有益的，但是会损害分类器学习，当使用解耦训练策略时，也会对分类器学习有害，但可以忽略不计。[![知乎](https://img.shields.io/badge/知乎-white?logo=zhihu)](https://zhuanlan.zhihu.com/p/419911014)

**[3]** (**ICCV 2021**) FASA: Feature Augmentation and Sampling Adaptation for Long-Tailed Instance Segmentation 长尾实例分割的特征增强和采样自适应https://arxiv.org/abs/2102.12867

**[4]** (**ICVGIP'21**) Feature Generation for Long-tail Classification 长尾分类的特征生成 https://arxiv.org/abs/2111.05956

**[5]** (**ECCV2020**) Feature Space Augmentation for Long-Tailed Data 用于长尾数据的特征空间增广 https://arxiv.org/abs/2008.03673 这篇文章也可以放在特征迁移学习中

**[6]**（**CVPR 2020**）Deep Representation Learning on Long-tailed Data: A Learnable Embedding Augmentation Perspective 为尾部类别构造云、用“特征云”来充实尾部类的方法 https://arxiv.org/abs/2002.10826 **备注:** 和[1]基本一样，而且这篇论文研究似乎更加深入，[1]有抄袭的现象吗？这篇论文也可以被划分进入特征迁移学习的类别。

**[7]**（**ECCV WorkShop 2020**）Remix: Rebalanced Mixup 重新平衡的Mixup https://arxiv.org/pdf/2007.03943v3.pdf

## 特征迁移学习 Feature transfer learning
**[1]** (**CVPR 2019**) Feature Transfer Learning for Face Recognition With Under-Represented Data 针对欠表示数据人脸识别的特征迁移学习https://ieeexplore.ieee.org/document/8953809/citations#citations

**[2]** (**CVPR 2020**) Deep Representation Learning on Long-tailed Data: A Learnable Embedding Augmentation Perspective 基于长尾数据的深度表示学习:一种可学习的嵌入增强视角 https://arxiv.org/abs/2002.10826 为尾部类别构造云、用“特征云”来充实尾部类的方法

**[3]** (**ECCV 2020**) Feature Space Augmentation for Long-Tailed Data 用于长尾数据的特征空间增广 https://arxiv.org/abs/2008.03673 这篇文章也可以放在特征迁移学习中

**[4]** (**cvpr 2021**) RSG: A Simple but Effective Module for Learning Imbalanced Datasets 一个简单但有效的学习不平衡数据集的模块https://arxiv.org/abs/2106.09859

**[5]** (**CVPR 2020**) M2m: Imbalanced Classification via Major-to-minor Translation:除了特征层面的头到尾的转移，头尾转换(M2m)提出通过基于扰动的优化，将头类样本转换为尾类样本来增加尾类。通过基于扰动的优化，将头类样本翻译成尾类样本，这与对抗性攻击本质上相似。转换后的尾类样本将被用来构建一个更平衡的训练集进行模型训练。https://arxiv.org/abs/2004.00431

**[6]** (**ICCV 2021**) GistNet: a Geometric Structure Transfer Network for Long-Tailed Recognition  一种用于长尾识别的几何结构迁移网络:在分类器层面进行头到尾的转移。通过用头类相对较大的分类器几何信息来增强尾类的分类器权重，GIST能够获得更好的尾部类性能。https://arxiv.org/pdf/2105.00131.pdf

**[7]** (似乎未正式发表) Label-Aware Distribution Calibration for Long-tailed Classification 标签感知分布校准用于长尾分类 https://arxiv.org/abs/2111.04901

**[8]** (**ICML 2021 WorkShop**)Class-Distribution-Aware Calibration for Long-Tailed Visual Recognition 长尾视觉识别的类分布感知校准 https://arxiv.org/abs/2109.05263

**[9]** (**ICMI 2015**) Sharing Representations for Long Tail Computer Vision Problems 长尾计算机视觉问题的共享表示 https://dl.acm.org/doi/10.1145/2818346.2818348

## 集成学习 
**[1]**（**CVPR 2022**）Trustworthy Long-Tailed Classification 值得信赖的长尾分类 https://arxiv.org/abs/2111.09030

**[2]** (**ICLR 2021**) Long-tailed Recognition by Routing Diverse Distribution-Aware Experts 基于路由多样性分布感知专家的长尾识别 https://arxiv.org/pdf/2010.01809.pdf

**[3]** (**arxiv 2021**) Test-Agnostic Long-Tailed Recognition by Test-Time Aggregating Diverse Experts with Self-Supervision 具有自我监督的测试时间聚合不同专家的测试无关长尾识别： 这片文章应该是目前的SOTA  https://arxiv.org/abs/2107.09249

## logit Adiustment
**[1]** (**AAAI 2022**) Adaptive Logit Adjustment Loss for Long-Tailed Visual Recognition 长尾视觉识别的自适应Logit调整损失https://arxiv.org/pdf/2104.06094.pdf

**[2]** (**CVPR 2022**) Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment 通过高斯扰动调整logit的方法(构造“特征云”)，解决长尾分类中各类样本严重不平衡的问题 https://www.techrxiv.org/articles/preprint/Long-tailed_Visual_Recognition_via_Gaussian_Clouded_Logit_Adjustment/17031920/1

## 其他方法 Other methods
**[1]** (**CVPR 2021**) Memory-based Jitter: Improving Visual Recognition on Long-tailed Data with Diversity In Memory 基于内存的抖动：提高对具有内存多样性的长尾数据的视觉识别 https://arxiv.org/abs/2008.09809
<br />备注：这篇论文将尾部类别的历史特征保存下来，以实现和头部类别在数量上的平衡

**[2]**（**ACM MM 2021**）Long-tailed Distribution Adaptation 长尾分布适应 https://arxiv.org/pdf/2110.02686.pdf

**[3]** (2021年提交到arxiv) Calibrating Class Activation Maps for Long-Tailed Visual Recognition 校准类激活映射的长尾视觉识别 https://arxiv.org/abs/2108.12757

## 标准数据集
**CIFAR10-LT**, **CIFAR100-LT**, **ImageNet-LT**, **iNaturalist**, **Places-LT**

### Get in touch
[![GitHub](https://img.shields.io/badge/GitHub-grey?logo=github)](https://github.com/mayanbiao1234)
[![知乎](https://img.shields.io/badge/知乎-white?logo=zhihu)](https://www.zhihu.com/people/ma-yan-biao-73)


