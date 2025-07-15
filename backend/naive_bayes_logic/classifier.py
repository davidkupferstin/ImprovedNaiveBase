from naive_bayes_logic.tools import Tools
import pandas as pd
import numpy as np


class Classifier:
    """
    Implements a Naive Bayes classifier for categorical data,
    using precomputed conditional probabilities.
    """

    def __init__(self, train_df, percentage_of_values, customer_values=None):
        """
        Initialize the classifier.
        Args:
            train_df (pd.DataFrame): Training dataframe.
            percentage_of_values (dict): Nested dict of conditional probabilities.
            customer_values (dict, optional): Feature values for prediction.
        """
        self.train_df = train_df
        self.classification = self.train_df.iloc[:,
                              -1].value_counts()  # Returns the types of departments and how many of each there are.
        self.percentage_of_values = percentage_of_values
        self.customer_values = customer_values
        self.class_results = pd.Series(dtype='float64')
        self.is_testing = False  # This flag is now less relevant for API usage, but kept for consistency

    def predict(self, row_dict=None):
        """
        Perform prediction for a given set of feature values.
        Args:
            row_dict (dict, optional): Feature values to predict on.
                If None, use self.customer_values.
        """
        if row_dict:
            self.customer_values = row_dict
            self.is_testing = True

        if self.customer_values is None:
            raise ValueError("Customer values not set for prediction.")

        for class_name in self.classification.index:
            prior = self.classification[class_name] / self.train_df.shape[0]
            probs = []
            for column, value in self.customer_values.items():
                # Ensure the column exists in percentage_of_values for the current class
                if column not in self.percentage_of_values[class_name]:
                    # Handle cases where a feature column might not have been in training data for this class
                    # This might indicate an issue with the input or training data,
                    # but for robustness, we can assign a small probability or skip.
                    # For Naive Bayes, it's safer to assign a small non-zero probability.
                    prob = 1 / (self.classification[class_name] + 1)  # Laplace smoothing for unseen feature
                else:
                    prob = self.percentage_of_values[class_name][column].get(value, 0)
                    if prob == 0:
                        # Apply Laplace smoothing if the specific value was not seen for this class
                        prob = 1 / (self.classification[class_name] + 1)
                probs.append(prob)

            total_prob = prior * np.prod(probs)
            self.class_results[class_name] = total_prob

    def get_prediction(self):
        """
        Get the class prediction result.
        Returns:
            dict: Dictionary with 'prediction' key of the class with max probability
                  and 'full_results' with all class probabilities.
        """
        if self.class_results.empty:
            raise ValueError("Prediction has not been performed yet.")

        prediction = self.class_results.idxmax()

        # No printing to terminal in API context
        # if not self.is_testing:
        #     target_column = Tools.get_the_target_column(self.train_df)
        #     readable_target = Tools.split_camel_case(target_column)
        #     print(f'The prediction of whether to {readable_target} this time is {prediction}')

        return {
            "prediction": prediction,
            "full_results": self.class_results.to_dict()  # Convert Series to dict for JSON serialization
        }
