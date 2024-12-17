import pandas as pd
import numpy as np
import json
import pickle
import joblib
import warnings
from TextPreprocessor import TextPreprocessor as OpenPrep
from data_cleaner import DataCleaner
from combine_data import combine_raw_data
from close_preprocessor import Preprocessor as ClosePrep

# Suppress warnings for cleaner output
warnings.simplefilter(action = 'ignore', category = FutureWarning)

# Load pre-trained models and configurations
def load_configurations():
    with open('column_name_mapping.json', 'r') as file:
        column_name_mapping = json.load(file)
        
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
        
    with open('selected_close_features.txt', 'r') as file:
        selected_close_features = [line.strip() for line in file]
    
    with open('trained_columns.pkl', 'rb') as file:
        trained_columns = pickle.load(file)
        
    loaded_model = joblib.load('logreg_model.pkl')
    
    return column_name_mapping, vectorizer, selected_close_features, trained_columns, loaded_model

# Data cleaning function
def clean_data(data, column_name_mapping):
    # Drop unwanted columns
    columns_to_drop = ['Full Name:', 'Username', 'Timestamp'] # For the sample csv
    data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1, inplace=True)
    
    # Rename columns based on mapping
    data = data.rename(columns=column_name_mapping)
    
    # Clean data (handles both structured and unstructured parts)
    cleaner = DataCleaner()
    data, struc, unstruc, application = cleaner.clean_data(data, rename_columns=True)
    
    return struc, unstruc

# TF-IDF preprocessing
def preprocess_tfidf(unstruc, vectorizer):
    tfidf_data = pd.DataFrame()
    
    # Apply TF-IDF transformation to each unstructured column
    for column in unstruc.columns:
        try:
            X = vectorizer.transform(unstruc[column].astype(str))
            feature_names = [f"{column}_{feature}" for feature in vectorizer.get_feature_names_out()]
            tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names)
            tfidf_data = pd.concat([tfidf_data, tfidf_df], axis=1)
        except Exception as e:
            print(f"Error processing column '{column}': {e}")
    
    return tfidf_data

# Close-ended question preprocessing
def preprocess_close(struc):
    preprocessor = ClosePrep()
    struc = preprocessor.preprocess_close_ended(struc)
    return struc

# Open-ended response preprocessing
def preprocess_unstructured(unstruc):
    preprocessor = OpenPrep()
    # Preprocess each response in the unstructured data
    unstruc = unstruc.applymap(preprocessor.preprocess_response)
    return unstruc

# Combine all data (structured and unstructured)
def combine_data(struc, tfidf_data):
    combined_responses = combine_raw_data(struc=struc, tfidf_data=tfidf_data)
    return combined_responses

# Filter columns for final model input
def filter_columns_for_model(combined_responses, selected_close_features, trained_columns):
    # Load selected close features from the file
    with open('selected_close_features.txt', 'r') as file:
        selected_close_features = [line.strip() for line in file]
    
    # Separate TF-IDF and structured data columns
    tfidf_columns = [col for col in combined_responses.columns if 'application_open_' in col]
    struc_columns = [col for col in combined_responses.columns if col not in tfidf_columns]
    
    # Get the relevant TF-IDF columns from the combined dataframe
    real_time_tfidf_columns = [col for col in tfidf_columns if col in combined_responses.columns]
    relevant_tfidf = combined_responses[real_time_tfidf_columns]
    
    # Get the relevant structured (close-ended) columns
    real_time_close_columns = [col for col in selected_close_features if col in combined_responses.columns]
    relevant_close = combined_responses[real_time_close_columns]
    
    # Combine the relevant close and TF-IDF columns
    final_modeling = pd.concat([relevant_close.reset_index(drop=True), relevant_tfidf.reset_index(drop=True)], axis=1)
    
    # Ensure that the final dataframe has the same columns as the trained model
    for col in trained_columns:
        if col not in final_modeling.columns:
            final_modeling[col] = 0  # Add missing columns with 0
    
    # Reorder the columns to match the trained model
    final_modeling = final_modeling[trained_columns]
    return final_modeling
    
# Model prediction
def make_predictions(final_modeling, model):
    # Make predictions using the trained model
    y_pred_real_time = model.predict(final_modeling)
    y_prob_real_time = model.predict_proba(final_modeling)[:, 1]
    
    # Output predictions
    for i, pred in enumerate(y_pred_real_time):
        print(f"Prediction {i + 1}: {'Approved' if pred == 1 else 'Disapproved'}")
    
    return y_pred_real_time, y_prob_real_time

# Main function to process data and make predictions
def process_and_predict(data):
    # Load configurations and models
    column_name_mapping, vectorizer, selected_close_features, trained_columns, model = load_configurations()
    
    # Data cleaning and preprocessing
    struc, unstruc = clean_data(data, column_name_mapping)

    # Preprocess open-ended data (apply response preprocessing)
    unstruc = preprocess_unstructured(unstruc)
    
    # TF-IDF processing
    tfidf_data = preprocess_tfidf(unstruc, vectorizer)
    
    # Close-ended question processing
    struc = preprocess_close(struc)
    
    # Combine data
    combined_responses = combine_data(struc, tfidf_data)
    
    # Filter columns for final model input
    final_modeling = filter_columns_for_model(combined_responses, selected_close_features, trained_columns)
    
    # Make predictions
    y_pred_real_time, y_prob_real_time = make_predictions(final_modeling, model)
    
    return y_pred_real_time, y_prob_real_time