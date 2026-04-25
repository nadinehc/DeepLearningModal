- Random guesser accuracy : 1/32 = 3%

# Data analysis
- check to see if data is balanced (same number of examples in each class ?)
- compare train, test, val datasets

# Track A - closed world
- produce training data with random rotations, crops, ...
- use masked self attention to encode time
- treat it as a simple classification : 3D deep convolution
- Idée : prendre des kernels identités, les flatten pour les transformer en token, et appliquer l’attention masquée / cross attention frame 1 ⇔ frame 2 ⇔ frame 3 ⇔ frame 4

# Track B - open world
- use pretrained model
- finetune on this dataset
- vision language model : prompt a vision language model to describe images then prompt a language model to classify images based on this description


---
# Articles : 
- https://proceedings.mlr.press/v139/bertasius21a/bertasius21a-supp.pdf
- https://openaccess.thecvf.com/content_CVPR_2020/papers/Materzynska_Something-Else_Compositional_Action_Recognition_With_Spatial-Temporal_Interaction_Networks_CVPR_2020_paper.pdf
- https://arxiv.org/pdf/1711.11248

---

# 24/04 Questions
- target accuracies on models ?
- typical depth ? typical number of parameters ?
- learning rate ? batch size ?
=> try different values

- how to limit overfitting ?
    - if the number of parameters is too big, the model will likely overfit
    - 

- architectures tested 
    - convolution + attention
    - 3D convolution

- une fois qu'on a sélectionné la meilleure architecture, et avant de submit la solution, on peut réentraîner sur train + validation