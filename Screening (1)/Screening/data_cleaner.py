import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self):
        self.updates = {
            'Housing Ownership': {'Own': {'Landlord Pet Permission': 'None'}},
            'Employment Status': {'No': {'Employment Details': 'None'}},
            'Gated Community Status': {'Yes': {'Pet Safety Assurance': 'None'}},
            'Pet Living Environment': {'Indoors': {'Outdoor Setup Details': 'None'}},
            'Children in Household': {'No': {'Children Ages and Pet Interaction': 'None'}},
            'Household Pet Allergies': {'No': {'Allergy Management': 'None'}},
            'Pet Transfer Willingness': {'Yes': {'New Pet Transfer Reason': 'None'}},
            'Follow_Up Permission': {'Yes': {'Unavailable Contact Reason': 'None'}},
            'Interest in Adoption': {'No': {'Adoption Reason': 'Not interested'}},
            'Previous Pet Vaccinations Status': {'Not vaccinated': {'Previous Pet Vaccination Management': 'No Vaccine'}},
            'Previous Pet Spay/Neuter Status': {'Not yet spayed or neutered': {'Previous Pet Spay/Neuter Importance': 'No Response'}},
            'Spay/Neuter Willingness': {'No': {'New Pet Spay/Neuter Plan': 'No Response'}}
        }
        self.none_variations = ['none', 'NONE', 'n/a', 'N/A', '']

    def apply_condition_updates(self, data: pd.DataFrame) -> pd.DataFrame:
        for condition_col, condition_dict in self.updates.items():
            for condition_val, target_cols in condition_dict.items():
                for target_col, target_value in target_cols.items():
                    if condition_col in data.columns:
                        data.loc[data[condition_col] == condition_val, target_col] = target_value
        return data

    def replace_null_variations(self, data: pd.DataFrame) -> pd.DataFrame:
        # Replace any variations of 'None' and NaN with 'None'
        data.replace(self.none_variations + [np.nan], 'None', inplace=True)
        return data

    def rename_and_separate_columns(self, data: pd.DataFrame) -> tuple:
        questions = pd.Series(data.columns)

        # Define question types
        quest_type = {
            'close': [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 33, 35, 37, 39, 41]
        }
        quest_type['open'] = [i for i in range(len(questions)) if i not in quest_type['close']]

        # Define task types
        task_type = {
            'application': [i for i in range(len(questions))]
        }

        # Helper functions for naming
        def get_task_type(index):
            for task, indices in task_type.items():
                if index in indices:
                    return task
            return 'unknown'

        def get_quest_type(index):
            for quest, indices in quest_type.items():
                if index in indices:
                    return quest
            return 'unknown'

        # Rename columns
        new_column_names = []
        for i, col in enumerate(questions):
            task = get_task_type(i)
            quest = get_quest_type(i)
            new_name = f"{task}_{quest}_{i}"
            new_column_names.append(new_name)
        data.columns = new_column_names

        # Separate DataFrames
        structured_df = data.iloc[:, quest_type['close']].copy()
        unstructured_df = data.iloc[:, quest_type['open']].copy()
        application_df = data.iloc[:, task_type['application']].copy()

        return data, structured_df, unstructured_df, application_df

    def clean_data(self, data: pd.DataFrame, rename_columns: bool = False) -> tuple:
        # Handle nulls and condition updates globally
        data = self.replace_null_variations(data)
        data = self.apply_condition_updates(data)

        if rename_columns:
            data, struc, unstruc, application = self.rename_and_separate_columns(data)
            return data, struc, unstruc, application
        else:
            return data