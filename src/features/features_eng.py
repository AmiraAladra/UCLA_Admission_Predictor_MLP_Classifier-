import pandas as pd
import logging

def Create_dummy_variables(df):
    """
    Creates dummy variables for categorical columns and splits
    the dataset into features (X) and target (y).

    Parameters:
        df (pd.DataFrame): Cleaned DataFrame including 'Admit_Chance'.

    Returns:
        x (pd.DataFrame): Feature matrix with dummy variables.
        y (pd.Series): Target vector (Admit_Chance).
    """
    try:
        logging.info("Creating dummy variables...")

        # Ensure the target column exists
        if 'Admit_Chance' not in df.columns:
            raise KeyError("Target column 'Admit_Chance' not found in DataFrame.")

        # Ensure specified columns exist
        for col in ['University_Rating', 'Research']:
            if col not in df.columns:
                raise KeyError(f"Expected column '{col}' not found in DataFrame.")

        # Create dummy variables for categorical features
        clean_data = pd.get_dummies(df, columns=['University_Rating', 'Research'], dtype='int')

        # Separate features and target
        x = clean_data.drop(['Admit_Chance'], axis=1)
        y = clean_data['Admit_Chance']

        logging.info("Dummy variable creation successful. Shape: X=%s, y=%s", x.shape, y.shape)
        return x, y

    except Exception as e:
        logging.error("Error in Create_dummy_variables: %s", str(e), exc_info=True)
        raise
