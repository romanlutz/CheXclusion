# CheXclusion: Fairness gaps in deep chest X-ray classifiers

This is the code for 'CheXclusion: Fairness gaps in deep chest X-ray classifiers' (https://arxiv.org/abs/2003.00827) paper accepted in 'PSB 2021'.

In this paper, we examine the extent to which state-of-the-art deep learning classifiers trained to yield diagnostic labels from X-ray images are biased with respect to protected attributes, such as patient sex, age, race, and insurance type as a proxy for socioeconomic status. In particular, we examine the differences in true positive rate (TPR) across different subgroups per attributes. A high TPR disparity indicates that sick members of a protected subgroup would not be given correct diagnoses---e.g., true positives---at the same rate as the general population, even in an algorithm with high overall accuracy. 

We train convolution neural networks to predict 14 diagnostic labels in 3 prominent public chest X-ray datasets: MIMIC-CXR, Chest-Xray8, CheXpert, as well as a multi-site aggregation of all those datasets (ALLData). 

This code is also a good learning resource for researcher/students interested in training multi-label medical image pathology classifiers. 


@article{CheXclusion_2020,

  title={CheXclusion: Fairness gaps in deep chest X-ray classifiers},
  
  author={Seyyed-Kalantari, Laleh and Liu, Guanxiong and McDermott, Matthew and Chen, Irene and Marzyeh, Ghassemi},
  
  BOOKTITLE = {Pacific Symposium on Biocomputing},
  
  year={2021}
}
