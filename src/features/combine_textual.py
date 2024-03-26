# -*- coding: utf-8 -*-
import logging
import pandas as pd
import re


# Define a function to combine the textual features of each row
def combine_features(row, feature_list):
    """Combine the textual features into a single column."""
    logger = logging.getLogger(__name__)
    logger.info('Combining textual features into a single column\n')

    row['CombinedText'] = ''

    for feature in feature_list:
        if feature == 'AGE':
            if pd.notna(row[feature]):
                # Combine 'AGE' as direct text description
                row['CombinedText'] = ' '.join([row['CombinedText'], f'{int(row[feature])}_year_old'])

                # Combine 'AGE' as age group
                if row['AGE'] < 2: row['CombinedText'] = ' '.join([row['CombinedText'], 'Infant'])
                elif row['AGE'] >= 2 and row['AGE'] < 4: row['CombinedText'] = ' '.join([row['CombinedText'], 'Toddler'])
                elif row['AGE'] >= 4 and row['AGE'] < 12: row['CombinedText'] = ' '.join([row['CombinedText'], 'Child'])
                elif row['AGE'] >= 12 and row['AGE'] < 20: row['CombinedText'] = ' '.join([row['CombinedText'], 'Teenager'])
                elif row['AGE'] >= 20 and row['AGE'] < 40: row['CombinedText'] = ' '.join([row['CombinedText'], 'Adult'])
                elif row['AGE'] >= 40 and row['AGE'] < 60: row['CombinedText'] = ' '.join([row['CombinedText'], 'Middle_Aged'])
                elif row['AGE'] >= 60: row['CombinedText'] = ' '.join([row['CombinedText'], 'Senior'])
            continue

        if feature == 'SEX':
            if isinstance(row[feature], str):
                row['CombinedText'] = ' '.join([row['CombinedText'], row[feature]])
            continue

        # Comine `USETOBAC` if the patient is a current tobacco user
        if feature == 'USETOBAC':
            if row[feature] == 'Current':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Tobacco_User'])
            continue

        # Combine `visitReason` and rule out the non-relevant reasons
        if feature == 'MAJOR':
            if pd.notna(row[feature]):
                if row['MAJOR'].contains('Chronic problem') & row['MAJOR'].contains('routine'):
                    row['CombinedText'] = ' '.join([row['CombinedText'], 'Routine_chronic_problem'])
                elif row['MAJOR'].contains('Chronic problem') & row['MAJOR'].contains('flare-up'):
                    row['CombinedText'] = ' '.join([row['CombinedText'], 'Flare_up_chronic_problem'])
                elif row['MAJOR'].contains('New problem'):
                    row['CombinedText'] = ' '.join([row['CombinedText'], 'New_problem (less than 3 months onset)'])
                elif row['MAJOR'].contains('Preventive care'):
                    row['CombinedText'] = ' '.join([row['CombinedText'], 'Preventive_care'])
                elif row['MAJOR'].contains('Pre/Post-surgery'):
                    row['CombinedText'] = ' '.join([row['CombinedText'], 'Pre_or_Post_surgery'])
                elif row['MAJOR'].contains('Acute problem'):
                    row['CombinedText'] = ' '.join([row['CombinedText'], 'Acute_problem'])
            continue

        if feature in ['RFV1', 'RFV2', 'RFV3']:
            if isinstance(row[feature], str) and row[feature] not in [
                'Problems, complaints, NEC',
                'Patient unable to speak English',
                'Patient (or spokesperson) refused care',
                'Entry of "none" or "no complaint"',
                'Inadequate data base',
                'Illegible entry'
            ]:
                row['CombinedText'] = ' '.join([row['CombinedText'], row[feature]])
            continue

        # Combine `vitalSigns` as Textual Descriptions
        # `HTIN`, `WTLB` and `BMI` are combined as a single description of the patient's weight condition.
        if feature == 'BMI':
            if pd.notna(row[feature]):
                if row['BMI'] < 18.5: row['CombinedText'] = ' '.join([row['CombinedText'], 'Underweight'])
                elif row['BMI'] >= 18.5 and row['BMI'] < 25: row['CombinedText'] = ' '.join([row['CombinedText'], 'Normal_weight'])
                elif row['BMI'] >= 25 and row['BMI'] < 30: row['CombinedText'] = ' '.join([row['CombinedText'], 'Overweight'])
                elif row['BMI'] >= 30: row['CombinedText'] = ' '.join([row['CombinedText'], 'Obesity'])
            continue

        # `TEMPF` are combined as a single description of the patient's temperature condition.
        if feature == 'TEMPF':
            if pd.notna(row[feature]):
                if row['TEMPF'] < 95: row['CombinedText'] = ' '.join([row['CombinedText'], 'Hypothermia'])
                elif row['TEMPF'] >= 95 and row['TEMPF'] < 99: row['CombinedText'] = ' '.join([row['CombinedText'], 'Normal_temperature'])
                elif row['TEMPF'] >= 99 and row['TEMPF'] < 100: row['CombinedText'] = ' '.join([row['CombinedText'], 'Low_grade_fever'])
                elif row['TEMPF'] >= 100 and row['TEMPF'] < 103: row['CombinedText'] = ' '.join([row['CombinedText'], 'Fever'])
                elif row['TEMPF'] >= 103: row['CombinedText'] = ' '.join([row['CombinedText'], 'Hyperpyrexia'])
            continue
        
        # `BPSYS` are combined as a single description of the patient's blood pressure condition.
        if feature == 'BPSYS':
            if pd.notna(row[feature]):
                if row['BPSYS'] < 90: row['CombinedText'] = ' '.join([row['CombinedText'], 'Hypotension'])
                elif row['BPSYS'] >= 90 and row['BPSYS'] < 120: row['CombinedText'] = ' '.join([row['CombinedText'], 'Normal_blood_pressure'])
                elif row['BPSYS'] >= 120 and row['BPSYS'] < 140: row['CombinedText'] = ' '.join([row['CombinedText'], 'Prehypertension'])
                elif row['BPSYS'] >= 140: row['CombinedText'] = ' '.join([row['CombinedText'], 'Hypertension'])
            continue

        # `BPDIAS` are combined as a single description of the patient's blood pressure condition.
        if feature == 'BPDIAS':
            if pd.notna(row[feature]):
                if row['BPDIAS'] < 60: row['CombinedText'] = ' '.join([row['CombinedText'], 'Low_diastolic_blood_pressure'])
                elif row['BPDIAS'] >= 60 and row['BPDIAS'] < 90: row['CombinedText'] = ' '.join([row['CombinedText'], 'Normal_diastolic_blood_pressure'])
                elif row['BPDIAS'] >= 90 and row['BPDIAS'] < 110: row['CombinedText'] = ' '.join([row['CombinedText'], 'High_diastolic_blood_pressure'])
                elif row['BPDIAS'] >= 110: row['CombinedText'] = ' '.join([row['CombinedText'], 'Hypertension'])
            continue

        # Convert and combine `presentSymptomsStatus` and rule out 'NO' and 'NONE' symptoms
        if feature == 'ARTHRTIS':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Arthritis'])
            continue

        if feature == 'ASTHMA':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Asthma'])
            continue

        if feature == 'CANCER':
            if row[feature] == 'Yes' :
                if pd.notna(row['CASTAGE']): row['CombinedText'] = ' '.join([row['CombinedText'], '_'.join((row['CASTAGE'] + ' Cancer').split())])
                else: row['CombinedText'] = ' '.join([row['CombinedText'], 'Cancer'])
            continue

        if feature == 'CEBVD':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Cerebrovascular_disease'])
            continue

        if feature == 'CHF':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Congestive_heart_failure'])
            continue

        if feature == 'CRF':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Chronic_renal_failure'])
            continue

        if feature == 'COPD':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Chronic_obstructive_pulmonary_disease'])
            continue
            
        if feature == 'DEPRN':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Depression'])
            continue
            
        if feature == 'DIABETES':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Diabetes'])
            continue
            
        if feature == 'HYPLIPID':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Hyperlipidemia'])
            continue
            
        if feature == 'HTN':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Hypertension'])
            continue
            
        if feature == 'IHD':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Ischemic_heart_disease'])
            continue
            
        if feature == 'OBESITY':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Obesity'])
            continue
            
        if feature == 'OSTPRSIS':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Osteoporosis'])
            continue
            
        if feature == 'NOCHRON':
            if row[feature] == 'Yes':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'No_chronic_condition'])
            continue
            
        if feature == 'DMP':
            if row[feature] == 'Not enrolled':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Not_enrolled_in_a_disease_management_program'])
            elif row[feature] == 'Currently enrolled':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Currently_enrolled_in_a_disease_management_program'])
            elif row[feature] == 'Ordered/advised to enroll at this visit':
                row['CombinedText'] = ' '.join([row['CombinedText'], 'Ordered_or_advised_to_enroll_in_a_disease_management_program'])
            continue

        # Combine `physicianDiagnoses` and rule out 'PROBABLE, QUESTIONABLE, OR RULE OUT' diagnoses
        if feature in ['DIAG1', 'DIAG2', 'DIAG3']:
            diag_id = re.search(r'DIAG(\d)', feature).group(1)
            prdiag = f'PRDIAG{diag_id}'
            if isinstance(row[feature], str) & ((row[prdiag] == 'No') | ('not probable' in row[prdiag])):
                row['CombinedText'] = ' '.join([row['CombinedText'], row[feature]])
            continue

    return row['CombinedText'].strip()



if __name__ == '__main__':
    import argparse

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description='Combine the textual features into a single column')
    parser.add_argument(
        "--dataframe", type=str, required=True,
        help='Path to the dataframe'
    )
    parser.add_argument(
        "--feature_list", type=str, required=True,
        help='List of features to be combined'
    )
    args = parser.parse_args()

    # Check if all the features are present in the dataframe
    for feature in args.feature_list:
        assert feature in args.dataframe.columns, f'Feature `{feature}` is not present in the dataframe'

    # Load the dataframe
    df = pd.read_csv(args.dataframe)