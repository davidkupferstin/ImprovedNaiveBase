import pandas as pd # Added for type hinting

class Tools:
    """
    Utility class with static helper methods.
    """
    @staticmethod
    def split_camel_case(text):
        """
        Split camel case string into space separated words.
        Args:
            s (str): CamelCase string.
        Returns:
            str: Space separated string.
        """
        if not text:
            return ""
        words = []
        current = text[0]
        for char in text[1:]:
            if char.isupper():
                words.append(current)
                current = char
            else:
                current += char
        words.append(current)
        return " ".join(words)

    @staticmethod
    def get_the_target_column(df: pd.DataFrame) -> str:
        """
        Get the last column name of a dataframe, assumed as the target column.
        Args:
            df (pd.DataFrame): Dataframe to inspect.
        Returns:
            str: Name of the target column.
        """
        if df.empty:
            raise ValueError("DataFrame is empty, cannot determine target column.")
        return df.columns[-1]
