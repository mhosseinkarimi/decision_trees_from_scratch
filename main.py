from data.dataloader import DataLoader
from models.decision_tree import DecisionTree
from utils.ui import UI
from utils.metrics import accuracy
from sklearn.model_selection import train_test_split, KFold
import numpy as np


if __name__ == "__main__":
    # Opening
    ui = UI()

    # Loading data
    dataloader = DataLoader(ui.train_path, ui.test_path, ui.feature_names)

    if ui.load_option == 1:
        data = dataloader.load_from_csv(
            ui.load_test, ui.train_path, ui.test_path)
    elif ui.load_option == 2:
        data = dataloader.load_from_file(ui.load_test)
    else:
        dataloader.create_csv("./train.csv", "./test.csv")
        data = dataloader.load_from_csv(ui.load_test, "./train.csv", "./test.csv")

    if ui.load_test:
        train_data, test_data = data
    else:
        train_data = data

    # Initialize the Decision Tree
    tree = DecisionTree(max_length=ui.max_length, target_label=ui.target_label)

    if ui.kfold:
        train_accs = []
        val_accs = []
        test_accs = []
        kf = KFold(ui.n_folds, shuffle=True)
        fold = 1

        for train_index, val_index in kf.split(train_data):
            train_samples, val_samples = train_data.loc[train_index], train_data.loc[val_index]

            # training
            tree.train(train_samples, 0.01)

            # train accuracy
            pred, true = tree.predict(train_data)
            currnent_train_acc = accuracy(pred, true)
            train_accs.append(currnent_train_acc)
            print(f"Train Accuracy in fold #{fold}: {currnent_train_acc:.3f}")

            # validation
            pred, true = tree.predict(val_samples)

            # validation accuracy
            currnent_test_acc = accuracy(pred, true)
            val_accs.append(currnent_test_acc)
            print(
                f"Validatoin Accuracy in fold #{fold}: {currnent_test_acc:.3f}")

            fold += 1

        # test accuracy
        test_acc = accuracy(tree.predict(test_data))

        train_acc = np.mean(train_accs)
        val_acc = np.mean(val_accs)

        print(f"Train average accuracy wiht {ui.n_folds}-fold CV: {train_acc}")
        print(
            f"Validation average accuracy wiht {ui.n_folds}-fold CV: {val_acc}")
        print(f"Test average accuracy wiht {ui.n_folds}-fold CV: {test_acc}")

    else:

        train_accs = []
        test_accs = []
        for i in range(ui.train_repeat):
            train_samples, val_samples = train_test_split(
                train_data, train_size=ui.train_split_prc, shuffle=True)
            # training
            tree.train(train_samples, 0.01)

            # train accuracy
            pred, true = tree.predict(train_data)
            currnent_train_acc = accuracy(pred, true)
            train_accs.append(currnent_train_acc)
            print(f"Train Accuracy in trial #{i+1}: {currnent_train_acc:.3f}")

            # test accuracy
            test_pred, test_true = tree.predict(test_data)
            currnent_test_acc = accuracy(test_pred, test_true)
            test_accs.append(currnent_test_acc)
            print(f"Test Accuracy in trial #{i+1}: {currnent_test_acc:.3f}")

        train_acc = np.mean(train_accs)
        test_acc = np.mean(test_accs)

        print(
            f"Train average accuracy with {ui.train_repeat} repeats: {train_acc:.3f}")
        print(
            f"Test average accuracy with {ui.train_repeat} repeats: {test_acc:.3f}")
