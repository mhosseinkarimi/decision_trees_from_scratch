from typing import List
from utils.InformationGain import InformationGain
import pandas as pd
import numpy as np


class Node(InformationGain):
    def __init__(self, target_label: str = 'label', index: int = 0):
        super.__init__(target_label)
        self.main_feature = None
        self.index = index

    # Train methods

    def set_condition(self, samples: pd.DataFrame) -> None:
        """This method is used for training process to determine the most informative feature.

        Args:
            samples (pd.DataFrame): Data samples
        """
        target_idx = samples.columns.get_loc(super.target_label)
        feature_cols = samples.columns.delete(target_idx)

        feature_igs = []
        for feature in feature_cols:
            feature_igs.append(super.information_gain(samples, feature))

        feature_igs = np.array(feature_igs)

        self.main_feature = feature_igs[np.argmax(feature_igs)]

    def calculate_loss(self, samples: pd.DataFrame) -> np.float64:
        """Calculates the entropy of the input samples

        Args:
            samples (pd.DataFrame): Input data samples

        Returns:
            np.float64: batch entropy
        """
        return super.entropy(samples)

    def create_branches(self, samples: pd.DataFrame) -> List:
        """This method creates the branches resulted by training the node.

        Args:
            samples (pd.DataFrame): Input data samples

        Returns:
            List: Partitions of data
        """
        if self.main_feature:
            split_data = super.split(samples, self.main_feature)
            return split_data
        else:
            raise(TypeError("The node hasn't been trained."))