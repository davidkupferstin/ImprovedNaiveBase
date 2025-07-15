import pandas as pd # Added for type hinting and potential DataFrame operations

class InformationCleaning:
    """
    Class to perform cleaning operations on a dataframe such as
    converting boolean columns to strings and cleaning column names.
    """
    def __init__(self, info_df):
        """
        Initialize with a dataframe to clean.
        Args:
            info_df (pd.DataFrame): The dataframe to be cleaned.
        """
        self._cleaning_info = info_df.copy()

    def clean_all(self):
        """
        Perform all cleaning steps on the dataframe.
        """
        self._convert_bool_columns_to_str()
        self._clean_column_names()

    def _convert_bool_columns_to_str(self):
        """
        Convert all boolean columns in the dataframe to string type.
        """
        bool_columns = self._cleaning_info.select_dtypes(include='bool').columns
        for col in bool_columns:
            self._cleaning_info[col] = self._cleaning_info[col].astype(str)

    def _clean_column_names(self):
        """
        Clean column names by stripping whitespace and replacing
        dashes and spaces with underscores. Rename 'class' column to 'label'.
        """
        self._cleaning_info.columns = (
            self._cleaning_info.columns
            .str.strip()
            .str.replace('-', '_', regex=False)
            .str.replace(' ', '_', regex=False)
        )
        # Check if 'class' column exists before renaming
        if 'class' in self._cleaning_info.columns:
            self._cleaning_info.rename(columns={'class': 'label'}, inplace=True)

    def get_dataframe(self):
        """
        Get the cleaned dataframe.
        Returns:
            pd.DataFrame: The cleaned dataframe.
        """
        return self._cleaning_info
