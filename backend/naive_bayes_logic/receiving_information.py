import pandas as pd
from naive_bayes_logic.information_cleaning import InformationCleaning

class ReceivingInformation:
    """
    Class responsible for loading data from a CSV file, cleaning it,
    and managing train/test split of the dataset.
    """
    def __init__(self, path):
        """
        Initialize ReceivingInformation by reading CSV and cleaning data.
        Args:
        path (str): Path to the CSV file.
        """
        df = pd.read_csv(path)
        cleaner = InformationCleaning(df)
        cleaner.clean_all()
        self._df = cleaner.get_dataframe()
        self._train_df = None
        self._test_df = None

    def split_train_test(self):
        """
        Split the cleaned dataframe into training (70%) and testing (30%) datasets.
        """
        # Ensure reproducibility with a fixed random_state
        self._train_df = self._df.sample(frac=0.7, random_state=42)
        self._test_df = self._df.drop(self._train_df.index)

    def get_dataframe(self):
        """
        Get the cleaned full dataframe.
        Returns:
        pd.DataFrame: Cleaned dataframe.
        """
        return self._df

    def get_train_df(self):
        """
        Get the training dataframe after split.
        Raises:
            ValueError: If train/test split has not been done yet.
        Returns:
            pd.DataFrame: Training dataframe.
        """
        if self._train_df is None:
            raise ValueError("Train/Test split has not been performed yet. Please call split_train_test() first.")
        return self._train_df

    def get_test_df(self):
        """
        Same as the previous one for the testing dataframe
        """
        if self._test_df is None:
            raise ValueError("Train/Test split has not been performed yet. Please call split_train_test() first.")
        return self._test_df
