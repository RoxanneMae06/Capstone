import pandas as pd

def combine_raw_data(struc, tfidf_data):
    """
    Combines the raw close-ended data and the raw TF-IDF features.

    Args:
        struc_df (pd.DataFrame): Raw close-ended data.
        tfidf_data (pd.DataFrame): Raw TF-IDF data.

    Returns:
        pd.DataFrame: Combined DataFrame containing the raw close-ended and raw TF-IDF features.
    """
    combined_responses = pd.concat([struc, tfidf_data], axis=1)
    
    return combined_responses
