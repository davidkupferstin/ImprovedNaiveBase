from naive_bayes_logic.receiving_information import ReceivingInformation
from naive_bayes_logic.data_analyzer import DataAnalyzer
from naive_bayes_logic.model_testing import ModelTesting
from naive_bayes_logic.user_service import UserService
from naive_bayes_logic.classifier import Classifier
from naive_bayes_logic.tools import Tools
import pandas as pd  # Added for type hinting and potential DataFrame operations

# Global variables to store the trained model and related data
# In a real-world application, this would be persisted in a database or a more robust cache.
# For this example, we'll use in-memory storage.
trained_model_data = {
    "percentage_of_values": None,
    "train_df": None,
    "test_df": None,
    "full_df": None,
    "accuracy": None,
    "features": None,
    "target_column": None
}


def load_data_and_split(dataset_path):
    """
    Loads the dataset and splits it into training and testing sets.
    Args:
        dataset_path (str): Path to the CSV dataset.
    Returns:
        tuple: (train_df, test_df, full_df) — DataFrames for training, testing, and full original dataset.
    """
    info = ReceivingInformation(dataset_path)
    info.split_train_test()
    return info.get_train_df(), info.get_test_df(), info.get_dataframe()


def analyze_training_data(train_df, full_df):
    """
    Analyzes the training data to calculate conditional probabilities for each feature.
    Args:
        train_df (pd.DataFrame): The training set.
        full_df (pd.DataFrame): The full dataset before split.
    Returns:
        dict: A nested dictionary with class -> feature -> value -> probability.
    """
    analyzer = DataAnalyzer(train_df, full_df)
    analyzer.trainer()
    return analyzer.get_percentage_of_values()


def test_model_accuracy(test_df, percentage_of_values, train_df):
    """
    Evaluates and returns the model's accuracy on the test set.
    Args:
        test_df (pd.DataFrame): The test set.
        percentage_of_values (dict): Precomputed probabilities from the training data.
        train_df (pd.DataFrame): Training data (needed for Classifier initialization).
    Returns:
        float: Model accuracy.
    """
    examination = ModelTesting(test_df, percentage_of_values, train_df)  # Pass train_df
    examination.evaluate_model_accuracy()
    return examination.get_model_accuracy()


def collect_user_input(train_df, from_json_request):
    """
    Collects input values for prediction from an external JSON.
    Args:
        train_df (pd.DataFrame): Training data used for valid value checks.
        from_json_request (dict): Input values for prediction (non-interactive mode).
    Returns:
        dict: A dictionary of feature names and their values for prediction.
    """
    user_service = UserService(train_df)
    user_service.set_customer_values(from_json_request)
    return user_service.get_customer_values()


def make_prediction(train_df, percentage_of_values, customer_values):
    """
    Performs the prediction based on user input.
    Args:
        train_df (pd.DataFrame): Training data.
        percentage_of_values (dict): Conditional probabilities for each class.
        customer_values (dict): Input values for prediction.
    Returns:
        dict: Prediction result with predicted class and full probability distribution.
    """
    classifier = Classifier(train_df, percentage_of_values, customer_values)
    classifier.predict()
    prediction = classifier.get_prediction()
    return prediction


def train_model_workflow(dataset_path):
    """
    Main workflow to load data, train model, and store results.
    Args:
        dataset_path (str): Path to the CSV dataset.
    Returns:
        dict: A dictionary containing model metadata and accuracy.
    """
    train_df, test_df, full_df = load_data_and_split(dataset_path)
    percentage_of_values = analyze_training_data(train_df, full_df)
    accuracy = test_model_accuracy(test_df, percentage_of_values, train_df)

    target_column = Tools.get_the_target_column(train_df)
    feature_columns = [col for col in train_df.columns if col != target_column]

    # Store the trained model data globally
    trained_model_data["percentage_of_values"] = percentage_of_values
    trained_model_data["train_df"] = train_df
    trained_model_data["test_df"] = test_df
    trained_model_data["full_df"] = full_df
    trained_model_data["accuracy"] = accuracy
    trained_model_data["features"] = {
        col: train_df[col].astype(str).unique().tolist() for col in feature_columns
    }
    trained_model_data["target_column"] = target_column

    return {
        "message": "Model trained successfully!",
        "accuracy": accuracy,
        "features": trained_model_data["features"],
        "target_column": target_column
    }


def predict_workflow(customer_values):
    """
    Workflow to make a prediction using the currently trained model.
    Args:
        customer_values (dict): Input data for prediction.
    Returns:
        dict: Prediction results including predicted class and full probabilities.
    """
    if trained_model_data["percentage_of_values"] is None:
        raise ValueError("Model not trained yet. Please upload a dataset first.")

    train_df = trained_model_data["train_df"]
    percentage_of_values = trained_model_data["percentage_of_values"]

    # Validate customer_values against expected features
    expected_features = list(trained_model_data["features"].keys())
    if not all(f in customer_values for f in expected_features):
        raise ValueError(f"Missing features in input. Expected: {expected_features}")

    # Ensure input values are strings, matching how they are stored in the model
    customer_values_str = {k: str(v) for k, v in customer_values.items()}

    return make_prediction(train_df, percentage_of_values, customer_values_str)


def get_model_status():
    """
    Returns the current status of the trained model.
    """
    if trained_model_data["percentage_of_values"] is None:
        return {"status": "No model trained", "accuracy": None, "features": None, "target_column": None}

    return {
        "status": "Model trained",
        "accuracy": trained_model_data["accuracy"],
        "features": trained_model_data["features"],
        "target_column": trained_model_data["target_column"]
    }
