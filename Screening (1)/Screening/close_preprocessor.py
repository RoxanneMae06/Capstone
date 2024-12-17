import pandas as pd

class Preprocessor:
    def __init__(self):
        # Ordinal encoding mappings
        self.age_mapping = {
            '15-25 years old': 1,
            '26-35 years old': 2,
            '36-45 years old': 3,
            '46-55 years old': 4,
            '56-65 years old': 5,
            '66-75 years old': 6
        }
        self.spay_neuter_mapping = {
            'Spayed': 2,
            'Neutered': 2,
            'Not yet spayed or neutered': 1,
            'None': 0
        }
        self.vaccination_mapping = {
            'Fully Vaccinated': 3,
            'Vaccinated': 3,
            'Partially vaccinated': 2,
            'Not vaccinated': 1,
            'None': 0
        }

    def preprocess_close_ended(self, struc):
        # Apply ordinal encoding for Age Range
        if 'application_close_0' in struc.columns:
            struc.loc[:, 'application_close_0'] = struc['application_close_0'].map(self.age_mapping)

        # Apply mapping functions
        if 'application_close_27' in struc.columns:
            struc.loc[:, 'application_close_27'] = struc['application_close_27'].apply(
                lambda x: max(self.spay_neuter_mapping.get(s.strip(), 0) for s in x.split(';'))
            )
        if 'application_close_29' in struc.columns:
            struc.loc[:, 'application_close_29'] = struc['application_close_29'].apply(
                lambda x: max(self.vaccination_mapping.get(s.strip(), 0) for s in x.split(';'))
            )

        # One-Hot Encoding for specific columns
        struc = pd.get_dummies(struc, columns=[
            'application_close_1',
            'application_close_2',
            'application_close_19',
            'application_close_21',
            'application_close_23',
            'application_close_25',
        ], drop_first=True)

        # Binary encoding for specific columns
        binary_columns = [
            'application_close_3',
            'application_close_5',
            'application_close_9',
            'application_close_13',
            'application_close_15',
            'application_close_17',
            'application_close_33',
            'application_close_35',
            'application_close_37',
            'application_close_39',
            'application_close_41'
        ]

        for column in binary_columns:
            if column in struc.columns:
                struc.loc[:, column] = struc[column].map({'Yes': 1, 'No': 0})

        # Map 'Own' to 1 and 'Rent' to 0 for Housing Ownership
        if 'application_close_7' in struc.columns:
            struc.loc[:, 'application_close_7'] = struc['application_close_7'].map({'Own': 1, 'Rent': 0})

        # Map 'indoors' to 1 and 'outdoors' to 0 for Pet Living Environment
        if 'application_close_11' in struc.columns:
            struc.loc[:, 'application_close_11'] = struc['application_close_11'].map({'Indoors': 1, 'Outdoors': 0})

        return struc