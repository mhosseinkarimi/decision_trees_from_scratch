from typing import Any, List, Tuple
from utils.information_gain import InformationGain
import pandas as pd
import numpy as np
from scipy.stats import mode


class Node():
    """Node class represents the building block of tree data structure. 
    It stores the value for the main feature, which is the the decision making 
    condition of the node. Also, it stores the stage of the node so we can check 
    the node position in the tree and to enable us to limit the length of the tree.
    """

    def __init__(self, n_stage: int, target_label: str = 'label'):
        """This class needs the stage of the node and the label of the target data in the
        data samples' pandas DataFrame.

        Args:
            n_stage (int): The stage number 
            target_label (str, optional): The label of the target column of data
        """
        self.ig = InformationGain(target_label)
        self.target_label = target_label
        self.main_feature = None
        self.child_nodes = []
        self.label = None
        self.n_stage = n_stage

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

        self.main_feature = feature_cols[np.argmax(feature_igs)]

    def calculate_loss(self, samples: pd.DataFrame) -> np.float64:
        """Calculates the entropy of the input samples

        Args:
            samples (pd.DataFrame): Input data samples

        Returns:
            np.float64: batch entropy
        """
        return self.ig.entropy(samples)

    def create_nodes(self, samples: pd.DataFrame) -> zip:
        """This method creates the child resulted by training the parent node.

        Args:
            samples (pd.DataFrame): Input data samples

        Returns:
            A zip of data and corresponding child node
        """
        split_data = self.ig.split(samples, self.main_feature)

        for _ in range(len(split_data)):
            self.child_nodes.append(Node(self.n_stage+1, self.target_label))

        if len(split_data) == len(self.child_nodes):
            return zip(self.child_nodes, split_data)
        else:
            raise(IndexError(
                "The number of the nodes and the data partitions aren't equal"))

    def set_label(self, samples) -> None:
        """Sets the value of the label of the node. Should be called if training of the node is done

        Args:
            label (Any): The label that is applied to the node data samples
        """
        data_labels = samples[self.target_label]

        self.label = mode(data_labels, axis=None)[0].item()

    def split(self, samples: pd.DataFrame, condition: str) -> List:
        """This method splits the given data samples with respect to the categories of the condition making feature.

        Args:
            samples (pd.DataFrame): Data samples that are passed to the node.
            condition (str): the name of chosen feature for conditions of the split.

        Returns:
            list of pd.DataFrame: The list of split samples based on the condition feature.
        """
        split_df = []

        unique_vlues = list(set(samples[condition]))
        unique_vlues.sort()

        for value in unique_vlues:
            mask = samples[condition] == value
            split_df.append(samples[mask])

        return split_df

    def train_node(self, samples: pd.DataFrame) -> Any:
        """This method handles training each node by either creating child nodes or
        labeling the node as a final node or Leaf.

        Args:
            samples (pd.DataFrame): Data samples

        Returns:
            Any: Either a zip of child nodes and data splits or None
        """
        if self.main_feature:
            return self.create_nodes(samples)
        else:
            self.set_label(samples)
            return None

    def predict_node(self, samples: pd.DataFrame) -> Any:
        """This method handles prediction of data labels using trained nodes.

        Args:
            samples (pd.DataFrame): Data sample

        Returns:
            Any: Either A list of data splits or None
        """
        if self.main_feature:
            return self.split(samples, self.main_feature)
        elif self.label:
            return None

    def __repr__(self) -> str:
        """Deifining the representation of the data.

        Returns:
            str: The string representation of the data
        """
        return ("Node : {main feature | " + str(self.main_feature) + "\n" +
                "number of child nodes | " + str(len(self.child_nodes)) + "\n" +
                "Stage | " + str(self.n_stage) + "\n" +
                "Label | " + str(self.label) + "}")

    def __eq__(self, __o: object) -> bool:
        """Equality function. Checks the equality of two Node object

        Args:
            __o (object): The target object

        Returns:
            bool: True if two objects are equal and False they aren't.
        """
        return self.__hash__() == __o.__hash__()
