from naive_bayes_logic.tools import Tools
import pandas as pd  # Added for type hinting


class UserService:
    """
    Handles interaction with the user to collect input values for prediction.
    """

    def __init__(self, train_df):
        """
        Initialize with training dataframe to know possible feature values.
        Args:
            train_df (pd.DataFrame): The training dataframe.
        """
        self.train_df = train_df
        self.customer_values = {}

    def collect_customer_values(self):
        """
        This method is for interactive terminal use and will not be used in the API.
        It's kept for reference to the original logic.
        """
        target_column = Tools.get_the_target_column(self.train_df)
        readable_target = Tools.split_camel_case(target_column)
        # print(f'Get a prediction on whether to {readable_target} this time.')
        # print('Please enter values from the options as requested to receive a prediction.')
        for column in self.train_df.columns[:-1]:
            # print(column)
            # print(self.train_df[column].unique())
            # value = input('>')
            # while value not in self.train_df[column].astype(str).unique():
            #     print("Invalid value. Please choose from the list above.")
            #     value = input("> ").strip()
            # self.customer_values[column] = value
            pass  # No interactive input in API context

    def set_customer_values(self, values_dict):
        """
        Set customer values directly from a dictionary.
        Args:
            values_dict (dict): Dictionary of feature values.
        """
        # Basic validation: ensure all expected feature columns are present
        target_column = Tools.get_the_target_column(self.train_df)
        expected_features = [col for col in self.train_df.columns if col != target_column]

        if not all(feature in values_dict for feature in expected_features):
            missing_features = [f for f in expected_features if f not in values_dict]
            raise ValueError(f"Missing required features: {', '.join(missing_features)}")

        # Validate values against unique values from training data
        for column, value in values_dict.items():
            if column in expected_features:  # Only validate feature columns
                allowed_values = self.train_df[column].astype(str).unique()
                if str(value) not in allowed_values:
                    raise ValueError(
                        f"Invalid value '{value}' for feature '{column}'. Allowed values: {', '.join(allowed_values)}")

        self.customer_values = {k: str(v) for k, v in values_dict.items()}  # Ensure all values are strings

    def get_customer_values(self):
        """
        Get the currently stored customer values.
        Returns:
            dict: Customer feature values.
        """
        return self.customer_values
