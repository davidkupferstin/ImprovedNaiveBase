from naive_bayes_logic.classifier import Classifier
from naive_bayes_logic.tools import Tools
import pandas as pd  # Added for type hinting and potential DataFrame operations


class ModelTesting:
    """
    Class to test a trained model's accuracy on a test dataframe.
    """

    def __init__(self, test_df, percentage_of_values, train_df):  # Added train_df
        """
        Initialize with test data and precomputed feature value percentages.
        Args:
            test_df (pd.DataFrame): The testing dataframe.
            percentage_of_values (dict): Nested dictionary of feature probabilities.
            train_df (pd.DataFrame): The training dataframe, needed for Classifier initialization.
        """
        self.test_df = test_df
        self.percentage_of_values = percentage_of_values
        self.correct = 0
        self.accuracy = 0.0
        # Pass train_df to Classifier as it needs it to determine target column and class counts
        self.classifier = Classifier(train_df, self.percentage_of_values)

    def evaluate_model_accuracy(self):
        """
        Evaluate the model accuracy by predicting each row and
        comparing with actual labels. Stores the accuracy as a percentage.
        """
        total = len(self.test_df)
        # Ensure feature columns are correctly identified, excluding the target column
        label_column = Tools.get_the_target_column(self.test_df)
        features = [col for col in self.test_df.columns if col != label_column]

        for index, row in self.test_df.iterrows():  # Use iterrows for easier dict conversion
            row_dict = {feature: str(row[feature]) for feature in features}  # Convert values to string
            self.classifier.predict(row_dict)
            prediction_result = self.classifier.get_prediction()
            prediction = prediction_result['prediction']
            actual = str(row[label_column])  # Convert actual to string for consistent comparison

            if prediction == actual:
                self.correct += 1

        if total > 0:
            self.accuracy = self.correct / total * 100
        else:
            self.accuracy = 0.0  # Handle empty test_df case

    def get_model_accuracy(self):
        """
        Return the accuracy of the model on the test data.
        Returns:
            float: Accuracy percentage.
        """
        # No printing to terminal in API context
        # print(f"The accuracy is {self.accuracy:.0f}%")
        return self.accuracy
