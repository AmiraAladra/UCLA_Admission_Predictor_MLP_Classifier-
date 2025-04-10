import pandas as pd
import logging

def load_and_clean_data(data_path):
    """
    Loads and cleans the admission dataset.

    - Reads a CSV file from `data_path`.
    - Converts 'Admit_Chance' to binary (1 if >= 0.8).
    - Drops 'Serial_No' column.
    - Converts 'University_Rating' and 'Research' to categorical.

    Parameters:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        logging.info(f"Reading dataset from {data_path}")
        df = pd.read_csv(data_path)

        # Check required columns
        required_columns = ['Admit_Chance', 'Serial_No', 'University_Rating', 'Research']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col}")

        # Convert Admit_Chance to binary
        df['Admit_Chance'] = (df['Admit_Chance'] >= 0.8).astype(int)

        # Drop Serial_No
        df = df.drop(['Serial_No'], axis=1)

        # Convert categorical fields
        df['University_Rating'] = df['University_Rating'].astype('object')
        df['Research'] = df['Research'].astype('object')

        logging.info("Dataset loaded and cleaned successfully.")
        return df

    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        raise

    except KeyError as e:
        logging.error(f"Data cleaning error: {e}")
        raise

    except Exception as e:
        logging.error("Unexpected error during data loading/cleaning", exc_info=True)
        raise
