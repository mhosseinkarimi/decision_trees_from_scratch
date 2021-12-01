# Decision Tree from Scratch

---

**Mohammad Hossein Karimi**

**Email: mhusseinkarimi@aut.ac.ir**

**Github: https://github.com/mhosseinkarimi/**

**Project repository: https://github.com/mhosseinkarimi/decision_trees_from_scratch**

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
$ python main.py
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

# Results of Training

**Part 1.a.**
in the table below I have listed the test and train accuracy for tree with different sizes. All training experiments are coducted with the same default configurations :

- Decision tree was trained on %45 of train data
- Training was repeated 3 times and the average accuracy is reported

| Size of Tree | Train Accuracy | Test Accuracy |
| ------------ | -------------- | ------------- |
| 4            | 0.690          | 0.683         |
| 5            | 0.696          | 0.692         |
| 6            | 0.738          | 0.722         |
| 7            | 0.699          | 0.676         |

The results of training suggest that with increasing the size of the tree, model tends to start overfitting. It should be noted that since the data has 8 features the tree with the length of 8 is the tree that has revised all training features as a condition.

**Part 1.b.**
In the last section we inspected the effect of decision tree size and, got to the coclusion that too samll or too lage trees can underfitting or overfitting.
In this section we are experimenting with %45, %55, %65, %75, %100 of training data samples for each tree.

**Decision Tree with the Size of 4**
| Train Size | Train Accuray | Test Accuracy |
|----------------|---------------|-------------------|
| %45 | 0.690 | 0.683 |
| %55 | 0.667 | 0.651 |
| %65 | 0.664 | 0.633 |
| %75 | 0.658 | 0.638 |
| %100| 0.660 | 0.625 |

**Decision Tree with the Size of 5**
| Train Size | Train Accuray | Test Accuracy |
|----------------|---------------|-------------------|
| %45 | 0.696 | 0.692 |
| %55 | 0.687 | 0.670 |
| %65 | 0.705 | 0.655 |
| %75 | 0.688 | 0.639 |
| %100| 0.658 | 0.633 |

**Decision Tree with the Size of 6**
| Train Size | Train Accuray | Test Accuracy |
|----------------|---------------|-------------------|
| %45 | 0.738 | 0.722 |
| %55 | 0.714 | 0.692 |
| %65 | 0.720 | 0.665 |
| %75 | 0.723 | 0.673 |
| %100| 0.650 | 0.628 |

**Decision Tree with the Size of 7**
| Train Size | Train Accuray | Test Accuracy |
|----------------|---------------|-------------------|
| %45 | 0.699 | 0.676 |
| %55 | 0.750 | 0.683 |
| %65 | 0.734 | 0.690 |
| %75 | 0.726 | 0.668 |
| %100| 0.648 | 0.631 |

The results show that by increasing the size of training dataset for bigger and stronger models we can improve their performance. But with increasing the size of the tree we are increasing the chance of overfitting. For using bigger models with stronger structures we can increase the number of data and use regularizations if possible.