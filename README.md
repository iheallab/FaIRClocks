# FaIRClocks

<div align="center">
  <img src="imgs/Figure 1.png" alt="main">
  <figcaption>Figure 1. The overall conceptual flowchart of FaIRClocks</figcaption>
</div>


--------------------------------------------------------------------------------

[FaIRClocks: Fair and Interpretable Representation of the Clock Drawing Test for mitigating classifier bias against lower educational groups](https://www.researchsquare.com/article/rs-3398970/v1)

## Introduction

This repository presents the "FaIRClocks" model, a novel approach from our study focusing on the fair and interpretable assessment of cognitive abilities using clock drawing tests (CDT). Leveraging the [Relevance Factor Variational Autoencoder (RF-VAE) network](https://arxiv.org/abs/1902.01568), FaIRClocks analyzes clock drawings to predict cognitive scores, including the Mini-Mental State Examination (MMSE) total score, attention composite z-score (ATT-C), and memory composite z-score (MEM-C). 

Our findings revealed that standard unweighted classifiers erroneously conflated lower education with cognitive impairment, leading to a significant type I error rate in this demographic. To mitigate this, we introduced multiple fairness metrics, effectively balancing performance across varying education levels. The result is a robust model capable of accurately identifying attention and memory deficits in individuals, irrespective of their education background.

Please also check our previous works exploring the application of deep learning techniques in the clock drawing test for cognitive outcomes prediction:

- 2022, [Explainable semi-supervised deep learning shows that dementia is associated with small, avocado-shaped clocks with irregularly placed hands](https://www.nature.com/articles/s41598-023-34518-9)

- 2023, [Variational autoencoder provides proof of concept that compressing CDT to extremely low-dimensional space retains its ability of distinguishing dementia](https://www.nature.com/articles/s41598-022-12024-8)

## Run

**Preparation**

Request the dataset (please find more information in `data/README.md`) and save the data under `data/`.

```shell
pip install -r requirements.txt
```

**Run training**

```shell
cd code
sh run.sh 
```

```shell
--data_root "path/to/your/data"
--data "MMSE" # The type of cognitive outcomes (MMSE, ATT, MEM)
--data_condition # The type of clock drawing conditions (copy, command, combined)
--model # The type of classifiers (LR, SVM, XGB)
--fair # Store if you want to use bias mitigation methods
```

## Results


Before we mitigated bias, the specificity metrics for both `copy` and `command` conditions were $0$, as shown in orange in Figure 2. This indicated a tendency of the model to inaccurately categorize patients with lower education levels as cognitively impaired. The blue plots in the figure represent the outcomes following bias mitigation. After bias mitigation, we not only enhanced the specificity for the low-educated group but also preserved the performance levels for the high-educated group.

<table>
  <tr>
    <td><img src="imgs/Figure 2a.png" alt="resulta", width="426"></td>
    <td><img src="imgs/Figure 2a.png" alt="resulta", width="426"></td>
  </tr>
  <tr>
    <td><img src="imgs/Figure 2c.png" alt="resulta", width="426"></td>
    <td><img src="imgs/Figure 2d.png" alt="resulta", width="440"></td>
  </tr>
  <caption style="caption-side:bottom">Figure 2. Effect of bias mitigation in classifying cognitive impairment by thresholding MMSE and ATT-C, in people with lower education (less than 8 years) versus those with higher education (greater than or equal to 8 years)</caption>
</table>

## Citation
```
@article{zhang2023fairclocks,
  title={FaIRClocks: Fair and Interpretable Representation of the Clock Drawing Test for mitigating classifier bias against lower educational groups},
  author={Zhang, Jiaqing and Bandyopadhyay, Sabyasachi and Kimmet, Faith and Wittmayer, Jack and Khezeli, Kia and Libon, David J and Price, Catherine C and Rashidi, Parisa},
  year={2023}
}
```