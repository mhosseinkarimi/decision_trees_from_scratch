import pandas as pd


class DataLoader:
    def __init__(self,
                 train_path: str,
                 test_path: str,
                 feature_names: list,
                 ) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.feature_names = feature_names
        self.feature_names.insert(0, 'label')

    def _convert2df(self, file_path: str, columns: list) -> pd.DataFrame:
        """Converts the dataset to a pandas.DataFrame.

        Args:
            file_path (str): The path to the dataset file
            columns (list): List of the headers of DataFrame 

        Returns:
            pd.DataFrame: Dataset in DataFrame format
        """
        df = pd.DataFrame(columns=columns)

        with open(file_path) as f:
            for i, line in enumerate(f):
                line = line.strip().split(',')
                df.loc[i] = line
        return df

    def _tonumerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts categorical data features and labels to numerical data.

        Args:
            df (pd.DataFrame): The input DataFrame with categorical data

        Returns:
            pd.DataFrame: The output DataFrame with numerical data
        """
        # making a copy of the input DataFrame
        df_copy = df.copy()
        # number of columns of df
        n_col = df.shape[1]
        # number of rows of df
        n_rows = len(df)

        # converting categorical data to numerical (could it be faster?)
        for i in range(n_col):
            # separating unique values
            unique_values = list(set(df_copy.iloc[:, i]))
            unique_values.sort()
            uniques_dict = dict(
                zip(unique_values, list(range(len(unique_values)))))

            for j in range(n_rows):
                df_copy.iloc[j][i] = uniques_dict[df_copy.iloc[j][i]]

        return df_copy

    def create_csv(self, train_csv_path: str = 'data/train.csv', test_csv_path: str = 'data/test.csv') -> None:
        """Creates csv file of  the train and test dataset as DataFrame.

        Returns:
            Tuple: a tuple of train and test DataFrames
        """
        # converting train dataset to DataFrame
        train_df = self._convert2df(self.train_path, self.feature_names)
        # converting test dataset to DataFrame
        test_df = self._convert2df(self.test_path, self.feature_names)

        # converting categorical data to numerical
        train_df = self._tonumerical(train_df)
        test_df = self._tonumerical(test_df)

        # creating csv files
        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)

    def load_from_file(self, load_test: bool=True) -> pd.DataFrame:
        """Loads train and test (if load_tset is True) as DataFrames

        Args:
            load_test (bool, optional): Determines to whether load test data or not. Defaults to True.

        Returns:
            pd.DataFrame: One [or two] pd.DataFrames representing train [and test] data
        """
        # converting train dataset to DataFrame
        train_df = self._convert2df(self.train_path, self.feature_names)
        # converting test dataset to DataFrame
        test_df = self._convert2df(self.test_path, self.feature_names)

        # converting categorical data to numerical
        train_df = self._tonumerical(train_df)
        test_df = self._tonumerical(test_df)

        if load_test:
            return train_df, test_df
        
        return train_df

    def load_from_csv(self, load_test: bool = True, train_csv_path: str = './train.csv', test_csv_path: str = './test.csv') -> pd.DataFrame:
        """Load the data from csv file

        Args:
            load_test (bool, optional): determines to whether load test data. Defaults to True.
            train_csv_path (str, optional): The path to tain csv file. Defaults to './train.csv'.
            test_csv_path (str, optional): The path to test csv file. Defaults to './test.csv'.

        Returns:
            pd.DataFrame: Train [and Test] data
        """
        train_df = pd.read_csv(train_csv_path)
        if load_test:
            test_df = pd.read_csv(test_csv_path)
            return train_df, test_df
        
        return train_df