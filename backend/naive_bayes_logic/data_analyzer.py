from naive_bayes_logic.tools import Tools
import pandas as pd  # Added for type hinting and potential DataFrame operations


class DataAnalyzer:
    """
    Analyzes the training dataframe to compute the conditional probabilities
    of feature values per class label for a Naive Bayes classifier.
    """

    def __init__(self, train_df, info_df):
        """
        Initialize with training data and information dataframe.
        Args:
            train_df (pd.DataFrame): The training dataframe.
            info_df (pd.DataFrame): Dataframe containing feature information.
        """
        self.train_df = train_df
        self.info_df = info_df
        self.percentage_of_values = {}

    def trainer(self):
        """
        Calculate the percentage of each feature value per class.
        Stores the results in self.percentage_of_values as a nested dictionary:
        {class_label: {feature_column: pd.Series of value percentages}}.
        """
        target_col = Tools.get_the_target_column(self.train_df)
        # Exclude the target column from feature columns
        feature_cols = [col for col in self.info_df.columns if col != target_col]

        # Collect all unique values for each feature across the entire training set
        all_unique_values = {col: self.train_df[col].astype(str).unique() for col in feature_cols}

        grouped = self.train_df.groupby(target_col)

        for class_name, group in grouped:
            self.percentage_of_values[class_name] = {}
            for col in feature_cols:
                # Convert column to string type before value_counts to ensure consistency
                counts = group[col].astype(str).value_counts()

                # Reindex with all possible unique values for the feature, filling missing with 0
                # This ensures that if a value doesn't appear in a specific class group,
                # it still has an entry with count 0, which is important for smoothing later.
                reindexed = counts.reindex(all_unique_values[col], fill_value=0)

                # Dividing each value by the frequency of the class as a whole
                self.percentage_of_values[class_name][col] = reindexed / group.shape[0]

    def get_percentage_of_values(self):
        """
        Get the computed percentages of feature values per class.
        Returns:
            dict: Nested dictionary of probabilities.
        """
        return self.percentage_of_values
