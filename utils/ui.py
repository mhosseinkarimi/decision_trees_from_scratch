from numpy import select
import os

class UI:
    def __init__(self):
        self.load_option = None
        self.train_path = None
        self.test_path = None
        self.feature_names = []
        self.load_test = None
        self.max_length = None
        self.target_label = 'label'
        self.data_split = None
        self.train_split_prc = None 
        self.train_repeat = None
        self.kfold = None
        self.n_folds = None
        # print("================================================")
        self.data_info()
        self.tree_info()
        self.train_info()
        # For Linux systems
        os.system("clear")

    def data_info(self):
        self.load_option = int(input("How do you want to load data? :\n1) Load csv file\n2) Load from original file\n3) create csv file then read from csv file\nchoose the option number [1,2,3]: "))

        print("Please enter the path to the data files")
        self.train_path = input("Train file path: ")
        self.test_path = input("Test file path: ")

        print("Please Enter feature names one at a time.")
        while(True):
            line = input("Enter \"finish\" for ending the process: ")
            if (line == 'finish'):
                break
            self.feature_names.append(line)

        try:
            self.load_test = {'y': True, 'n': False}[
                input("Do you want to load test data? [y/n]: ")]
        except:
            raise(KeyError("Invalid input please enter y or n"))

    def tree_info(self):
        self.max_length = int(
            input("Please enter the maximum length of tree: "))

    def train_info(self):
        self.kfold = {'y': True, 'n': False}[
            input("Do you want to perform K-fold corss validation? [y/n]: ")]
        if self.kfold:
            self.n_folds = int(input("Please enter the number of folds: "))

        self.data_split = {'y': True, 'n': False}[
            input("Do you want to split the train data? [y/n]: ")]
        if self.data_split:
            self.train_split_prc = float(
                input("What percent of the train dataset do you want to keep? [0 - 1]: "))
        self.train_repeat = int(input("How many times do you want to repeat training? "))
