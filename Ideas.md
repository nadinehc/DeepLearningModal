# Data analysis
- check to see if data is balanced (same number of examples in each class ?)
- compare train, test, val datasets

# Track A - closed world
- produce training data with random rotations, crops, ...
- use masked self attention to encode time
- treat it as a simple classification : 3D deep convolution

- hi nadine

# Track B - open world
- use pretrained model
- finetune on this dataset
- vision language model : prompt a vision language model to describe images then prompt a language model to classify images based on this description


---

# 24/04 Questions
- why are there duplicate classes ?