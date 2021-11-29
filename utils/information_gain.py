from typing import List
import pandas as pd
import numpy as np


class InformationGain:
    def __init__(self, target_label: str = 'label'):
        self.target_label = target_label

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

    def calculate_prob(self, samples: pd.DataFrame, feature_name: str) -> np.array:
        """Calculates the probabilty of each category in the chosen feature samples.

        Args:
            samples (pd.DataFrame): Data samples
            feature_name (str): The name of the feature and the labels of the column representing the feature in samples.

        Returns:
            np.array: List of the probabilities of each category in the feature
        """
        split_samples = self.split(samples, feature_name)

        n_samples = len(samples)
        n_class = len(split_samples)
        n_samples_in_class = [len(x) for x in split_samples]

        p_class = np.array([n_c/n_s for n_c,
                            n_s in zip(n_samples_in_class, n_class*[n_samples])])

        return p_class

    def entropy(self, samples: pd.DataFrame,) -> np.float64:
        """Calculates the entropy of a given data with arbitrary number of classes.

        Args:
            samples (pd.DataFrame): Given data samples used for calculating entropy

        Returns:
            entropy (np.float64): The entropy of the given datasamples.
        """
        entropy = 0

        p_class = self.calculate_prob(samples, self.target_label)

        entropy = -np.sum(p_class * np.log2(p_class))

        return entropy

    def conditional_entropy(self, samples: pd.DataFrame, feature_name: str) -> np.float64:
        """Calculates the value of the entropy of a set of samples fo a given feature.

        Args:
            samples (pd.DataFrame): Data samples 
            feature_name (str):  The name of the chosen feature as the prior condition

        Returns:
            np.float64: Conditional entropy of the samples with the feature as prior
        """
        split_samples = self.split(samples, feature_name)
        p_class = self.calculate_prob(samples, feature_name)
        entropies = np.array([self.entropy(x) for x in split_samples])

        conditional_entropy = np.sum(p_class * entropies)

        return conditional_entropy

    def information_gain(self, samples: pd.DataFrame, feature_name: str,) -> np.float64:
        """

        Args:
            samples (pd.DataFrame): Data samples
            feature_name (str): The name of the feature and the label of the column

        Returns:
            np.float64: The information gain of the feature
        """
        total_entropy = self.entropy(samples)

        class_entropy = self.conditional_entropy(samples, feature_name)

        information_gain = total_entropy - class_entropy

        return information_gain
