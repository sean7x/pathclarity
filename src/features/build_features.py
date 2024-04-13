# -*- coding: utf-8 -*-
import logging
import pandas as pd


def load_features(file_path):
    """Load the variables dictionary and return the features list"""
    import json


    with open(file_path, 'r') as f:
        variables = json.load(f)
    print(f'Variables keys: {variables.keys()}')

    # Defining the independent variables as features for classification
    features = \
        ['AGE', 'AGER', 'SEX', 'USETOBAC'] + variables['visitReason'] + ['PASTVIS'] + variables['vitalSigns'] \
        + variables['presentSymptomsStatus'] + variables['textFeature']
    
    print(f'Features: {features}')
    print(f'Number of Features: {len(features)}')

    return features


def load_rfv(file_path):
    """Load and clean the REASON FOR VISIT classification summary of codes"""
    rfv_df = pd.read_excel(file_path)

    # Split the 'CODE NUMBER' column into 'START' and 'END' columns
    rfv_df[['START', 'END']] = rfv_df['CODE NUMBER'].str.split('-', expand=True).astype(int)

    # Remove the leading and trailing whitespaces from `MODULE_1` and `MODULE_2` columns
    rfv_df['MODULE_1'] = rfv_df['MODULE_1'].str.strip()
    rfv_df['MODULE_2'] = rfv_df['MODULE_2'].str.strip()

    return rfv_df


def load_icd9cm(file_path):
    """Load the list of three-digit categories of ICD-9-CM"""
    icd9cm_df = pd.read_excel(file_path, dtype=str)

    return icd9cm_df


def build_features(df, features, rfv_df, icd9cm_df, category='CATEGORY_1'):
    """Preprocess and engineer the features, return X and y"""
    logger = logging.getLogger(__name__)
    logger.info('Preprocessing and engineering features\n')
    
    def get_module(code):
        """Find the `START` and `END` range, 
        and map the corresponding `MODULE_1` and `MODULE_2` to X_train as new columns `MODULE_1` and `MODULE_2`, 
        according to the value of `RFV1`, `RFV2`, and `RFV3` columns"""

        module = rfv_df.loc[(rfv_df['START'] <= code) & (rfv_df['END'] >= code), ['MODULE_1', 'MODULE_2']]
        if len(module) == 0:
            return pd.Series([pd.NA, pd.NA], index=['MODULE_1', 'MODULE_2'])
        else:
            return module.iloc[0]
    
    def get_icd9cm_3dcat(diag, prdiag, category=category):
        """Map the three-digit categories of ICD-9-CM to 'DIAG1', 'DIAG2', and 'DIAG3',
        if 'PRDIAG1', 'PRDIAG2', and 'PRDIAG3' are not 1 respectively"""

        try:
            if pd.notna(diag) and (pd.isna(prdiag) | prdiag != 1):
                if diag == 'V997-':
                    return 'No diagnosis/disease or healthy'
                else:
                    return icd9cm_df[icd9cm_df['3D_CODE'] == diag[:3]][category].values[0]
            else:
                return pd.NA
        except:
            print(f'Error: {diag}')
            print(f'Error: {prdiag}')
        
    def bin_age(age):
        if pd.isna(age): return pd.NA
        #if age < 2: return 'Infant'
        #elif age < 4: return 'Toddler'
        #elif age < 12: return 'Child'
        #elif age < 20: return 'Teenager'
        elif age < 20: return 'Child or Teenager'
        elif age < 40: return 'Adult'
        elif age < 60: return 'Middle Aged'
        else: return 'Senior'
    
    def bin_bmi(bmi):
        if pd.isna(bmi): return pd.NA
        elif bmi < 18.5: return 'Underweight'
        elif bmi < 25: return 'Normal weight'
        elif bmi < 30: return 'Overweight'
        else: return 'Obesity'
    
    def bin_tempf(tempf):
        if pd.isna(tempf): return pd.NA
        elif tempf < 95: return 'Hypothermia'
        elif tempf < 99: return 'Normal temperature'
        #elif tempf < 100: return 'Low grade fever'
        elif tempf < 103: return 'Fever'
        else: return 'Hyperpyrexia'
    
    def bin_bpsys(bpsys):
        if pd.isna(bpsys): return pd.NA
        elif bpsys < 90: return 'Hypotension'
        elif bpsys < 120: return 'Normal blood pressure'
        elif bpsys < 140: return 'Prehypertension'
        else: return 'Hypertension'

    def bin_bpdias(bpdias):
        if pd.isna(bpdias): return pd.NA
        elif bpdias < 60: return 'Low diastolic blood pressure'
        elif bpdias < 90: return 'Normal diastolic blood pressure'
        elif bpdias < 110: return 'High diastolic blood pressure'
        else: return 'Hypertension'

        
    X = df.loc[:, features].copy()

    # Bin the REASON FOR VISIT variables into RFV Modules
    X[['RFV1_MOD1', 'RFV1_MOD2']] = X['RFV1'].apply(
        lambda x: get_module(int(str(x)[:4])) if pd.notna(x) else pd.Series([pd.NA, pd.NA], index=['MODULE_1', 'MODULE_2'])
    )
    X[['RFV2_MOD1', 'RFV2_MOD2']] = X['RFV2'].apply(
        lambda x: get_module(int(str(x)[:4])) if pd.notna(x) else pd.Series([pd.NA, pd.NA], index=['MODULE_1', 'MODULE_2'])
    )
    X[['RFV3_MOD1', 'RFV3_MOD2']] = X['RFV3'].apply(
        lambda x: get_module(int(str(x)[:4])) if pd.notna(x) else pd.Series([pd.NA, pd.NA], index=['MODULE_1', 'MODULE_2'])
    )

    # Bin the AGE variable into AGE Groups
    X['AGE_GROUP'] = X['AGE'].apply(bin_age)

    # Bin the BMI variable into BMI Groups
    X['BMI_GROUP'] = X['BMI'].apply(bin_bmi)

    # Bin the TEMPF variable into TEMPF Groups
    X['TEMPF_GROUP'] = X['TEMPF'].apply(bin_tempf)

    # Bin the BPSYS variable into BPSYS Groups
    X['BPSYS_GROUP'] = X['BPSYS'].apply(bin_bpsys)

    # Bin the BPDIAS variable into BPDIAS Groups
    X['BPDIAS_GROUP'] = X['BPDIAS'].apply(bin_bpdias)

    # Handeling missing values in categorical features
    # Fill the missing values in the categorical features
    # with -9 for 'CASTAGE',
    # with -999 for 'USETOBAC', 'INJDET', 'MAJOR',
    # with -9 for 'RFV1', 'RFV2', 'RFV3'
    # with 'NA' for 'RFV1_MOD1', 'RFV2_MOD1', 'RFV3_MOD1', 'RFV1_MOD2', 'RFV2_MOD2', 'RFV3_MOD2',
    # with 'NA' for 'AGE_GROUP', 'BMI_GROUP', 'TEMPF_GROUP', 'BPSYS_GROUP', 'BPDIAS_GROUP'
    X.fillna({'CASTAGE': -9}, inplace=True)
    X.fillna({'USETOBAC': -999, 'INJDET': -999, 'MAJOR': -999}, inplace=True)
    X.fillna({'RFV1': -9, 'RFV2': -9, 'RFV3': -9}, inplace=True)
    X.fillna(
        {
            'RFV1_MOD1': 'NA', 'RFV2_MOD1': 'NA', 'RFV3_MOD1': 'NA',
            'RFV1_MOD2': 'NA', 'RFV2_MOD2': 'NA', 'RFV3_MOD2': 'NA',
            'AGE_GROUP': 'NA', 'BMI_GROUP': 'NA', 'TEMPF_GROUP': 'NA', 'BPSYS_GROUP': 'NA', 'BPDIAS_GROUP': 'NA'
        },
        inplace=True
    )

    # Employing the hierachical classifications of ICD-9-CM codes to prepare the target labels
    y = df.apply(lambda x: get_icd9cm_3dcat(x.DIAG1, x.PRDIAG1, category=category), axis=1)


    # Drop the rows from both X, y with NA in y
    non_missing_mask = y.notna()
    print(f'Number of available dependent samples: {non_missing_mask.sum()}')
    print()

    X = X.loc[non_missing_mask]
    print(f'X Shape: {X.shape}')

    y = y.loc[non_missing_mask]
    print(f'y with {category} Shape: {y.shape}')


    return X, y