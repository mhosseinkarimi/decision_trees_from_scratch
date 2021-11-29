from typing import Any, List
from information_gain import InformationGain
import pandas as pd
import numpy as np


class Node():
    def __init__(self, target_label: str = 'label'):
        self.ig = InformationGain(target_label)
        self.target_label = target_label
        self.main_feature = None
        self.child_nodes = []
        self.label = None

    # Train methods

    def set_condition(self, samples: pd.DataFrame) -> None:
        """This method is used for training process to determine the most informative feature.

        Args:
            samples (pd.DataFrame): Data samples
        """
        target_idx = samples.columns.get_loc(self.target_label)
        feature_cols = samples.columns.delete(target_idx)

        feature_igs = []

        for feature in feature_cols:
            feature_igs.append(self.ig.information_gain(samples, feature))
        
        feature_igs = np.array(feature_igs)

        self.main_feature = feature_igs[np.argmax(feature_igs)]

    def calculate_loss(self, samples: pd.DataFrame) -> np.float64:
        """Calculates the entropy of the input samples

        Args:
            samples (pd.DataFrame): Input data samples

        Returns:
            np.float64: batch entropy
        """
        return self.ig.entropy(samples)

    def create_nodes(self, samples: pd.DataFrame):
        """This method creates the branches resulted by training the node.

        Args:
            samples (pd.DataFrame): Input data samples

        Returns:
            List: Partitions of data
        """
        split_data = self.ig.split(samples, self.main_feature)

        for _ in range(len(split_data)):
            self.child_nodes.append(Node(self.target_label))

        return split_data

    def set_label(self, label: Any):
        """Sets the value of the label of the node. Should be called if training of the node is done

        Args:
            label (Any): The label that is applied to the node data samples
        """
        self.label = label

    def __repr__(self):
        return ("Node : {main feature | " + str(self.main_feature) + "\n" +
                "number of child nodes | " + str(len(self.child_nodes)) + "\n" +
                "Label | " + str(self.label) + "}")


if __name__ == "__main__":
    data = pd.DataFrame({'label': [0, 1, 1, 1, 1, 1, 0, 0], 'A': [
                        1, 0, 2, 1, 1, 2, 0, 0], 'B': [1, 0, 1, 1, 1, 0, 0, 1]})

    node = Node('label')

    print(f"Loss {node.calculate_loss(data)}")

    node.set_condition(data)

    node.create_nodes(data)

    node.set_label(0)

    print(node)
