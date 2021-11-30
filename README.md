# Decision Tree from Scratch
---
This projects aims to implement a decision tree structure and ID3 training algorithm.

The model is based on hierarchical relation between trees and node and ID3 algorithm guides the creation and growing of the tree using information gains as a metric.

Directory tree view of this project is presented below:

```
📦decision_trees_from_scratch
 ┣ 📂data
 ┃ ┗ 📜dataloader.py
 ┣ 📂models
 ┃ ┗ 📜tree.py
 ┣ 📂utils
 ┃ ┣ 📜information_gain.py
 ┃ ┣ 📜metrics.py
 ┃ ┣ 📜node.py
 ┃ ┣ 📜stack.py
 ┃ ┗ 📜ui.py
 ┣ 📜README.md
 ┗ 📜main.py
```

For training and prediction using decision tree you can run the following code in your terminal after navigating in the decision_trees_from_scratch folder on your local device:
```
main.py
```
After runnig the main module, UI process starts and you can enter your configuration.

Some of the parameters of the program that can be set are:
- load_from_csv
- load_from_csv
- train_path
- test_path
- feature_names
- load_test
- max_length
- target_label
- data_split
- train_split_prc
- train_repeat
- kfold
- n_folds


