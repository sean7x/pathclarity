# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
import altair as alt


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


def build_features(df, rfv_df, icd9cm_df, category='CATEGORY_1'):
    """Preprocess and engineer the features, return X and y"""
    logger = logging.getLogger(__name__)
    logger.info('Preprocessing and engineering features\n')
    
    def get_module(code):
        """Find the `START` and `END` range, 
        and map the corresponding `MODULE_1` and `MODULE_2` to X_train as new columns `MODULE_1` and `MODULE_2`, 
        according to the value of `RFV1`, `RFV2`, and `RFV3` columns"""

        module = rfv_df.loc[(rfv_df['START'] <= code) & (rfv_df['END'] >= code), ['MODULE_1', 'MODULE_2']]
        if len(module) == 0:
            return pd.Series([None, None], index=['MODULE_1', 'MODULE_2'])
        else:
            return module.iloc[0]
    
    def get_icd9cm_3dcat(diag, prdiag, category=category):
        """Map the three-digit categories of ICD-9-CM to 'DIAG1', 'DIAG2', and 'DIAG3',
        if 'PRDIAG1', 'PRDIAG2', and 'PRDIAG3' are not 1 respectively"""

        try:
            if pd.notna(diag) and diag != '-9' and (pd.isna(prdiag) | prdiag != 1):
                if diag == 'V997-':
                    return 'No diagnosis/disease or healthy'
                else:
                    return icd9cm_df[icd9cm_df['3D_CODE'] == diag[:3]][category].values[0]
            else:
                return None
        except:
            print(f'Error: {diag}')
            print(f'Error: {prdiag}')
        
    def bin_age(age):
        if pd.isna(age) or age == -9: return None
        #if age < 2: return 'Infant'
        #elif age < 4: return 'Toddler'
        #elif age < 12: return 'Child'
        #elif age < 20: return 'Teenager'
        elif age < 20: return 'Child or Teenager'
        elif age < 40: return 'Adult'
        elif age < 60: return 'Middle Aged'
        else: return 'Senior'
    
    def bin_bmi(bmi):
        if pd.isna(bmi): return None
        elif bmi < 18.5: return 'Underweight'
        elif bmi < 25: return 'Normal weight'
        elif bmi < 30: return 'Overweight'
        else: return 'Obesity'
    
    def bin_tempf(tempf):
        if pd.isna(tempf): return None
        elif tempf < 95: return 'Hypothermia'
        elif tempf < 99: return 'Normal temperature'
        #elif tempf < 100: return 'Low grade fever'
        elif tempf < 103: return 'Fever'
        else: return 'Hyperpyrexia'
    
    def bin_bpsys(bpsys):
        if pd.isna(bpsys): return None
        elif bpsys < 90: return 'Hypotension'
        elif bpsys < 120: return 'Normal blood pressure'
        elif bpsys < 140: return 'Prehypertension'
        else: return 'Hypertension'

    def bin_bpdias(bpdias):
        if pd.isna(bpdias): return None
        elif bpdias < 60: return 'Low diastolic blood pressure'
        elif bpdias < 90: return 'Normal diastolic blood pressure'
        elif bpdias < 110: return 'High diastolic blood pressure'
        else: return 'Hypertension'

        
    #X = df.loc[:, features].copy()

    # Bin the REASON FOR VISIT variables into RFV Modules
    df[['RFV1_MOD1', 'RFV1_MOD2']] = df['RFV1'].apply(
        lambda x: get_module(int(str(x)[:4])) if pd.notna(x) else pd.Series([None, None], index=['MODULE_1', 'MODULE_2'])
    )
    df[['RFV2_MOD1', 'RFV2_MOD2']] = df['RFV2'].apply(
        lambda x: get_module(int(str(x)[:4])) if pd.notna(x) else pd.Series([None, None], index=['MODULE_1', 'MODULE_2'])
    )
    df[['RFV3_MOD1', 'RFV3_MOD2']] = df['RFV3'].apply(
        lambda x: get_module(int(str(x)[:4])) if pd.notna(x) else pd.Series([None, None], index=['MODULE_1', 'MODULE_2'])
    )

    # Bin the AGE variable into AGE Groups
    df['AGE_GROUP'] = df['AGE'].apply(bin_age)

    # Bin the BMI variable into BMI Groups
    df['BMI_GROUP'] = df['BMI'].apply(bin_bmi)

    # Bin the TEMPF variable into TEMPF Groups
    df['TEMPF_GROUP'] = df['TEMPF'].apply(bin_tempf)

    # Bin the BPSYS variable into BPSYS Groups
    df['BPSYS_GROUP'] = df['BPSYS'].apply(bin_bpsys)

    # Bin the BPDIAS variable into BPDIAS Groups
    df['BPDIAS_GROUP'] = df['BPDIAS'].apply(bin_bpdias)

    # Handeling missing values in categorical features
    # Fill the missing values in the categorical features
    # with -9 for 'CASTAGE',
    # with -9 for 'USETOBAC', 'INJDET', 'MAJOR',
    # with -9 for 'RFV1', 'RFV2', 'RFV3'
    # with 'NA' for 'RFV1_MOD1', 'RFV2_MOD1', 'RFV3_MOD1', 'RFV1_MOD2', 'RFV2_MOD2', 'RFV3_MOD2',
    # with 'NA' for 'AGE_GROUP', 'BMI_GROUP', 'TEMPF_GROUP', 'BPSYS_GROUP', 'BPDIAS_GROUP'
    #df.fillna({'CASTAGE': -9}, inplace=True)
    df.fillna({'USETOBAC': -9, 'INJDET': -9, 'MAJOR': -9}, inplace=True)
    df.fillna({'RFV1': -9, 'RFV2': -9, 'RFV3': -9}, inplace=True)
    df.fillna(
        {
            'RFV1_MOD1': 'NA', 'RFV2_MOD1': 'NA', 'RFV3_MOD1': 'NA',
            'RFV1_MOD2': 'NA', 'RFV2_MOD2': 'NA', 'RFV3_MOD2': 'NA',
            'AGE_GROUP': 'NA', 'BMI_GROUP': 'NA', 'TEMPF_GROUP': 'NA', 'BPSYS_GROUP': 'NA', 'BPDIAS_GROUP': 'NA'
        },
        inplace=True
    )

    # Employing the hierachical classifications of ICD-9-CM codes to prepare the target labels
    df['DIAG1_CAT'] = df.apply(lambda x: get_icd9cm_3dcat(x.DIAG1, x.PRDIAG1, category=category), axis=1)
    df['DIAG2_CAT'] = df.apply(lambda x: get_icd9cm_3dcat(x.DIAG2, x.PRDIAG2, category=category), axis=1)
    df['DIAG3_CAT'] = df.apply(lambda x: get_icd9cm_3dcat(x.DIAG3, x.PRDIAG3, category=category), axis=1)

    # Drop the rows with non-relatative label
    # (SUPPLEMENTARY CLASSIFICATION OF FACTORS INFLUENCING HEALTH STATUS AND CONTACT WITH HEALTH SERVICES)
    df = df[
        df['DIAG1_CAT'] != 'SUPPLEMENTARY CLASSIFICATION OF FACTORS INFLUENCING HEALTH STATUS AND CONTACT WITH HEALTH SERVICES'
    ]

    return df


def combine_textual(row, features):
    """Convert and combine textual features into 'TEXT'"""
    logger = logging.getLogger(__name__)
    logger.info("Combining text features into 'TEXT'\n")

    row['TEXT'] = ''

    for feature in features:
        if feature == 'AGE':
            if pd.notna(row[feature]):
                # Combine 'AGE' as direct text description followed by 'AGE_GROUP'
                row['TEXT'] = ' '.join([
                    row['TEXT'], f'{int(row[feature])}_year_old',
                    '_'.join(row['AGE_GROUP'].split())
                ])
            continue

        if feature == 'SEX':
            if pd.notna(row[feature]):
                if row[feature] == 1:
                    row['TEXT'] = ', '.join([row['TEXT'], 'Female'])
                elif row[feature] == 0:
                    row['TEXT'] = ', '.join([row['TEXT'], 'Male'])
            continue

        # Comine `USETOBAC` if the patient is a current tobacco user
        if feature == 'USETOBAC':
            if row[feature] == 2:
                row['TEXT'] = ', '.join([row['TEXT'], 'Tobacco_User'])
            continue

        # Combine `visitReason` and rule out the non-relevant reasons
        if feature == 'INJDET':
            if pd.notna(row[feature]):
                if row[feature] == 1:
                    row['TEXT'] = ', '.join([row['TEXT'], 'Unintentional injury/poisoning'])
                elif row[feature] == 2:
                    row['TEXT'] = ', '.join([row['TEXT'], 'Intentional injury/poisoning'])
                elif row[feature] == 3:
                    row['TEXT'] = ', '.join([row['TEXT'], 'Injury/poisoning - unknown_intent'])
                elif row[feature] == 4:
                    row['TEXT'] = ', '.join([row['TEXT'], 'Adverse_effect of medical/surgical care or adverse_effect of medicinal drug'])
            continue

        if feature == 'MAJOR':
            if pd.notna(row[feature]):
                if row[feature] == 1:
                    row['TEXT'] = ', '.join([row['TEXT'], 'New problem'])
                elif row[feature] == 2:
                    row['TEXT'] = ', '.join([row['TEXT'], 'Chronic problem, routine'])
                elif row[feature] == 3:
                    row['TEXT'] = ', '.join([row['TEXT'], 'Chronic problem, flare_up'])
                elif row[feature] == 4:
                    row['TEXT'] = ', '.join([row['TEXT'], 'Pre-/Post-surgery'])
                elif row[feature] == 5:
                    row['TEXT'] = ', '.join([row['TEXT'], 'Preventive care (e.g. routine prenatal, well-baby, screening, insurance, general exams)'])
            continue

        if feature in ['RFV1', 'RFV2', 'RFV3']:
            if pd.notna(row[feature]) & (row[feature] != -9):
                # Combine 'RFVx_TEXT', followed by 'RFVx_MOD2', 'RFVx_MOD1'
                row['TEXT'] = ', '.join([row['TEXT'], row[f'{feature}_TEXT'], row[f'{feature}_MOD2'], row[f'{feature}_MOD1']])
            continue

        if feature in ['BMI', 'TEMPF', 'BPSYS', 'BPDIAS']:
            if pd.notna(row[feature]):
                # Combine 'feature_GROUP'
                row['TEXT'] = ', '.join([
                    row['TEXT'],
                    '_'.join(row[f'{feature}_GROUP'].split())
                ])
            continue

        # Convert and combine `presentSymptomsStatus` as direct text description
        if feature == 'ARTHRTIS':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Arthritis'])
            continue

        if feature == 'ASTHMA':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Asthma'])
            continue

        if feature == 'CANCER':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Cancer'])
            continue

        if feature == 'CEBVD':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Cerebrovascular_disease'])
            continue

        if feature == 'CHF':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Congestive_heart_failure'])
            continue

        if feature == 'CRF':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Chronic_renal_failure'])
            continue

        if feature == 'COPD':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Chronic_obstructive_pulmonary_disease'])
            continue

        if feature == 'DEPRN':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Depression'])
            continue

        if feature == 'DIABETES':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Diabetes'])
            continue

        if feature == 'HYPLIPID':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Hyperlipidemia'])
            continue

        if feature == 'HTN':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Hypertension'])
            continue

        if feature == 'IHD':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Ischemic_heart_disease'])
            continue

        if feature == 'OBESITY':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Obesity'])
            continue

        if feature == 'OSTPRSIS':
            if row[feature] == 1:
                row['TEXT'] = ', '.join([row['TEXT'], 'Osteoporosis'])
            continue

        # Combine `physicianDiagnoses` and rule out 'PROBABLE, QUESTIONABLE, OR RULE OUT' diagnoses
        if feature in ['DIAG1', 'DIAG2', 'DIAG3']:
            if pd.notna(row[feature]) and (pd.isna(row[f'PR{feature}']) | row[f'PR{feature}'] != 1):
                row['TEXT'] = ', '.join([row['TEXT'], row[f'{feature}_TEXT']])
            continue

    return row['TEXT'].strip()


def generate_topic_features(df, n_topics=10, n_top_words=10, transform=False, random_state=42):
    """Generate topic features (topic probabilities) from text features using LDA."""
    import spacy
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    logger = logging.getLogger(__name__)
    logger.info('Generating topic features (topic probabilities) from text features using LDA\n')

    # Preprocess the text features with Spacy
    nlp = spacy.load('en_core_web_sm')

    custom_stops = ['nos', 'oth', 'nec']
    for word in custom_stops:
        nlp.vocab[word].is_stop = True

    def preprocess_text(text):
        text = re.sub(r'\bdiabete\b', 'diabetes', text)
        text = re.sub(r'\banom\b', 'anomaly', text)
        text = re.sub(r'\bsho\b', 'shoulder', text)
        text = re.sub(r'\both\b', 'other', text)
        text = re.sub(r'\buns\b', 'unspecified', text)
        
        doc = nlp(text)
        filtered_tokens = [
            token.lemma_.lower() for token in doc
            if (not token.is_stop) and (not token.is_punct)
        ]
        return ' '.join(filtered_tokens)

    df['TEXT'] = df['TEXT'].apply(lambda row: preprocess_text(preprocess_text(row)))

    # Define the count vectorizer
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 1),
        max_features=1000,
        min_df=5,
        max_df=0.7,
    )
    tf = vectorizer.fit_transform(df['TEXT'])

    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='batch', n_jobs=-1, random_state=random_state)
    lda.fit(tf)

    # Define the function to display the top words for each topic
    def display_topics(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print(f'Topic {topic_idx}:')
            print(', '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print()
    
    display_topics(lda, vectorizer.get_feature_names_out(), n_top_words)

    # Define the topic features
    topics = lda.transform(tf)
    topic_features = [f'TOPIC_{i}' for i in range(topics.shape[1])]
    print(f'Topic Features: {topic_features}')
    topics = pd.DataFrame(topics, columns=topic_features, index=df.index)

    # Transform the topic features with PowerTransformer or Log transformation
    if transform == 'power':
        topics = np.sqrt(topics)
    elif transform == 'log':
        topics = np.log(topics + 0.0001)

    # Combine the topic features with df
    df = pd.concat([df, topics], axis=1)
    print(f'DataFrame Shape: {df.shape}')
    return df, vectorizer, tf, lda, topic_features


def generate_embeddings(df):
    """Add in sentence embeddings using BERT and pre-trained BiomedBERT model."""
    logger = logging.getLogger(__name__)
    logger.info('Generating sentence embeddings using BERT and pre-trained BiomedBERT model\n')

    from transformers import AutoTokenizer, AutoModel
    import torch

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device= 'mps'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)


    text = df['TEXT'].tolist()

    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        output = model(**encoded_input)
        sentence_embedding = output.last_hidden_state[:, 0, :]
        #sentence_embedding = output.pooler_output
        
        sentence_embedding.cpu().numpy()
    
    embed_features = [f'EMBED_{i}' for i in range(sentence_embedding.shape[1])]
    embed_df = pd.DataFrame(sentence_embedding, columns=embed_features)

    return pd.concat([df, embed_df], axis=1)


def chart(df, x, y, title, color=alt.value('steelblue'), width=480, height=320):
    """Pre-define chart function"""
    return alt.Chart(df).encode(
        x=x,
        y=y,
        color=color,
    ).properties(
    title=title,
    width=width,
    height=height,
).configure(
    axis=alt.AxisConfig(
        domain=False, # remove axis line
        ticks=False, # remove ticks
        labelAngle=0, # rotate labels
        labelColor='gray', # color of labels
        labelFontSize=10,
    ),
    font='Helvetica Neue',
    view=alt.ViewConfig(stroke=None), # remove border
)


def prepare_data(df, df_type, cleaned_data_path, figure_path, report_path, icd9cm_category='CATEGORY_1'):
    """Load and prepare the data for classification, return the processed DataFrame."""
    import os
    import pyLDAvis
    import pyLDAvis.lda_model

    # Load the variables dictionary and return the features list
    varaibles_path = os.path.join(cleaned_data_path, 'variables.json')
    features = load_features(varaibles_path)

    # Load and clean the REASON FOR VISIT classification summary of codes
    rfv_path = os.path.join('..', 'data', 'raw', 'RFV_codes_summary.xlsx')
    rfv_df = load_rfv(rfv_path)

    # Load the list of three-digit categories of ICD-9-CM
    icd9cm_path = os.path.join('..', 'data', 'raw', 'ICD9CM_3DCat.xlsx')
    icd9cm_df = load_icd9cm(icd9cm_path)

    icd9cm_category = icd9cm_category


    df = build_features(df, rfv_df, icd9cm_df, category=icd9cm_category)

    # Drop the rows from with NA in 'DIAG1_CAT'
    non_missing_mask = df['DIAG1_CAT'].notna()
    procd_df = df.loc[
        non_missing_mask,
        ['DIAG1_CAT'] + features + [
            'RFV1_MOD1', 'RFV1_MOD2', 'RFV2_MOD1', 'RFV2_MOD2', 'RFV3_MOD1', 'RFV3_MOD2',
            'AGE_GROUP', 'BMI_GROUP', 'TEMPF_GROUP', 'BPSYS_GROUP', 'BPDIAS_GROUP'
        ]
    ].copy()
    print(f'Number of available dependent samples: {non_missing_mask.sum()}')
    print()

    # Combine and preprocess textual features
    procd_df['TEXT'] = procd_df.apply(lambda x: combine_textual(x, features), axis=1)

    # Add in sentence embeddings using BERT and pre-trained BiomedBERT model
    #procd_df = .generate_embeddings(procd_df)

    # Add in topic feature (topic probabilities) using LDA
    transform = 'log'
    procd_df, vectorizer, tf, lda, topic_features = generate_topic_features(
        procd_df, n_topics=10, n_top_words=10, transform=transform
    )

    # Visualize the topics with pyLDAvis
    lda_vis = pyLDAvis.lda_model.prepare(lda, tf, vectorizer, mds='tsne')
    pyLDAvis.save_html(lda_vis, os.path.join(figure_path, f'{df_type}_lda_vis.html'))
    pyLDAvis.save_json(lda_vis, os.path.join(report_path, f'{df_type}_lda_vis.json'))

    # Plot the heat map of topic distributions among the labels in the dataset with Altair
    topic_df = procd_df[['DIAG1_CAT'] + topic_features].melt(id_vars='DIAG1_CAT', var_name='Topic', value_name='Probability')
    # Reverse the log transformation
    if transform == 'log':
        topic_df['Probability'] = np.exp(topic_df['Probability']) - 0.0001

    chart(
        df=topic_df,
        y='DIAG1_CAT:N',
        x='Topic:N',
        color='Probability:Q',
        title=f'Distribution of the Labels in the {df_type} Dataset',
    ).mark_rect().configure_axisY(
        labelLimit=500, title=None
    ).configure_axisX(
        labelAngle=45,
        title=None
    ).properties(width=300, height=500).save(os.path.join(figure_path, f'{df_type}_topic_distribution_heatmap.png'), ppi=300)

    # Plot the distribution and percentage of true labels
    chart(
        df=procd_df['DIAG1_CAT'].value_counts(normalize=True).reset_index(),
        x='DIAG1_CAT:N',
        y='proportion:Q',
        title=f'Distribution of the Labels in the {df_type} Dataset',
    ).mark_bar().configure_axisX(labelAngle=45, labelLimit=300, title=None).configure_axisY(title=None).save(os.path.join(figure_path, f'{df_type}_label_distribution.png'), ppi=300)
    
    return procd_df