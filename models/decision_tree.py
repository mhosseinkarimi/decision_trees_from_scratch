from typing import Any, Tuple
from utils.node import Node
import pandas as pd
import numpy as np

from utils.stack import Stack


class DecisionTree():
    """This class the tree data structure using hierarchical sturucture of the tree and 
    parent and child nodes."""

    def __init__(self, starter_node: Node = None, max_length: int = None, target_label: str = 'label') -> None:
        """
        Args:
            starter_node (Node, optional): The root node of the tree. Defaults to None.
            max_length (int, optional): Maximum allowed length of tree. Defaults to None.
            target_label (str, optional): The label of the target data column. Defaults to 'label'.
        """
        self.target_label = target_label
        self.starter_node = starter_node
        self.max_length = max_length

    def add_nodes(self, node: Node, samples: pd.DataFrame = None, added_nodes: Node = None) -> None:
        """This method is used to add child nodes to another node in the tree.
        Adding new nodes can be done through automatic process of adding the most iformative nodes 
        identified using ID3 or manually.

        Args:
            node (Node): Parent node
            samples (pd.DataFrame, optional): Data samples. Defaults to None.
            added_nodes (Node, optional): added nodes. Defaults to None.
        """
        if samples:
            node.create_nodes(samples)

        elif added_nodes:
            node.child_nodes.append(added_nodes)

        else:
            raise(TypeError("Both samples and added_nodes pramaeters are None Type"))

    def remove_nodes(self, src_node: Node, deleted_nodes: Node):
        """Removed the nodes passed the input.

        Args:
            src_node (Node): The source node (parent node)
            deleted_nodes (Node): Nodes to be deleted
        """
        for deleted_node in deleted_nodes:
            src_node.child_nodes.remove(deleted_node)
    
    def reset_tree(self):
        self.starter_node = None

    def train(self, samples: pd.DataFrame, epsilon: float = 0.01) -> None:
        """Trains the tree by initializing a root node and iterating over
        stacked nodes untill the conditions of train continuation are violated.

        Args:
            samples (pd.DataFrame): Data samples
            epsilon (float, optional): The minimum valus of the entropy for data in the node. Defaults to 0.01.
        """
        # initialize the decision tree
        if self.starter_node:
            self.reset_tree()
            
        self.starter_node = Node(1, self.target_label)

        # train sample and node pairs
        train_stack = Stack()
        train_stack.push((self.starter_node, samples))

        while(train_stack.size() > 0):
            stack_node, stack_samples = train_stack.pop()
            loss = stack_node.calculate_loss(stack_samples)
            if loss >= epsilon and stack_node.n_stage < self.max_length:
                stack_node.set_condition(stack_samples)

            node_pred = stack_node.train_node(stack_samples)

            if node_pred:
                for node_data_pair in node_pred:
                    train_stack.push(node_data_pair)

    def predict(self, samples: pd.DataFrame) -> Tuple:
        """Predicts the label for unknown data using the trained tree.

        Args:
            samples (pd.DataFrame): Data samples

        Returns:
            Tuple: A tuple containing the predicted labels and ground truth
        """

        if not self.starter_node:
            raise(TypeError("The decision tree hasn't been trained"))

        # prediction node data pair stack
        pred_stack = Stack()
        pred_stack.push((self.starter_node, samples))

        labels = pd.DataFrame({'predicted_labels': []})
        data = pd.DataFrame()

        while(pred_stack.size() > 0):
            stack_node, stack_samples = pred_stack.pop()

            node_predict = stack_node.predict_node(stack_samples)

            if node_predict:
                pred_data = zip(stack_node.child_nodes, node_predict)
                for node_data_pair in pred_data:
                    pred_stack.push(node_data_pair)
            else:
                pred_labels = pd.DataFrame(
                    {'predicted_labels': len(stack_samples) * [stack_node.label]})
                labels = pd.concat([labels, pred_labels], ignore_index=True)
                data = pd.concat([data, stack_samples], ignore_index=True)

        return labels.to_numpy(), data[self.target_label].to_numpy()

if __name__ == "__main__":
    train_csv_path = './train.csv'
    test_csv_path = './test.csv'

    train_samples = pd.read_csv(train_csv_path)
    test_samples = pd.read_csv(test_csv_path)

    tree = DecisionTree(max_length=5, target_label='label')

    tree.train(train_samples)

    labels, data = tree.predict(train_samples)

    print(np.mean(np.to_array(data) == np.to_array(labels)))
