PathClarity
==============================

PathClarity by Team Watson

## Introduction

Patients often experience unease when they must visit the hospital outpatient department, as they may lack clarity about their illness, the required examinations, and the associated expenses.

Machine learning with clinical data has long been in practice, offering substantial benefits.

While numerous studies have explored diagnostic and support pathways for specific ailments like depression and diabetes, the application of these technologies in creating interactive self-check platforms remains less explored.

Our project aims to:
- Could analyzing medical encounter data reveal distinct patient groups with similar healthcare needs?
- Could classification prediction on user-reported symptoms and health status enable the identification of potential treatment pathways in a statistical representation?

## Method

### Dataset
- Outpatient dataset of National Hospital Ambulatory Medical Care Survey (NHAMCS) with ICD-9-CM

### Preprocessing
- Merged: by consolidating relevant columns, resulting in a dataset with 150 features.
- Splitted into training (2006-2008), validation (2009), test (2010), and evaluation (2011) sets to maintain generalizability and adaptability to data shift.

### Feature Engineering
- Quantitative features: Binning or normalization based on stability and sensitivity requirements.
- Categorical features: One-hot encoding
- Text features: text extracted, integrated and transformed into topic features

### Machine Learning
 - Employed Random Forest Classifier to predict the classification of Diseases and Injuries


## Results
The specifics can be found in the concluding report (`./reports/final_report.pdf`).


## Conclusion
- Our machine learning analysis of NHAMCS outpatient data reveals key diagnostic pathways and supports, potentially improving patient understanding before medical consultations. 
- Despite time and data limitations, our promising results endorse the application of machine learning in healthcare, setting a foundation for future interactive self-check platforms.


## About Watson

The name of 'Watson' is adepted from the historical and famous 'Dr. Watson', a debugging tool included with the Microsoft Windows operating system since 1991.
You may find more about it following the Wikipedia. (https://en.wikipedia.org/wiki/Dr._Watson_(debugger))

## Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── cleaned        <- Merged and cleaned data for exploration.
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   │                     the creator's initials, and a short `-` delimited description.
    │   ├── 0-xt-parse_datasets.ipynb                             <- Load and parse the ASCII OPD data files and save to ./data/cleaned as Parquet files.
    │   ├── 1.0-xt-merge_datasets.ipynb                           <- Merge, clean and split the CSV files, with text as values, 
    │   │                                                            loaded with SPSS code, save to ./data/interim.
    │   ├── 1.1-xt-merge_clean_raw_datasets.ipynb                 <- Merge, clean and split the Parquet files, with survey codes as values,
    │   │                                                            save to ./data/interim.
    │   ├── 2.0-jj-stats_analysis.ipynb.ipynb                     <- Statistical analysis on the OPD data.
    │   ├── 3.1.1-jy-xt-kmeans_clustering.ipynb                   <- The clustering approach which try to group all the samples based on various features
    │   │                                                            to reveal distinct patient groups.
    │   ├── 3.2.1-jy-xt-classification_models.ipynb               <- Tryouts on different classification models and combination of features,
    │   │                                                            to predict the classification labels of ICD-9-CM.
    │   ├── 3.2.2-jy-xt-logistic_regression_grid_search.ipynb     <- Hyper parameters tuning with HalvingGridSearchCV for the Logistic Regression Classifer.
    │   ├── 3.2.3-jy-xt-random_forest_grid_search.ipynb           <- Hyper parameters tuning with HalvingGridSearchCV for the Random Forest Classifer.
    │   ├── 3.2.4-jy-xt-hgbct_grid_search.ipynb                   <- Hyper parameters tuning with HalvingGridSearchCV for the 
    │   │                                                            Histogram-Based Gradient Boosting Classifier.
    │   ├── 3.3-xt-model_final_evaluation.ipynb                   <- Evaluate the best model with the final evaluation dataset, and retrain with
    │   │                                                            all datasets concatenated to get the final model.
    │   ├── 4.1.1-jy-xt-user_input_matching_BiomedBERT.ipynb      <- The downstream task for the clustering approach, utilizing pre-trained BERT model,
    │   │                                                            trying to generate sentence embeddings and compute the similarities to the embeddings of
    │   │                                                            pre-defined patient groups to find the matching group.
    │   │                                                            
    │   ├── 5.0-xt-features_for_output.ipynb                      <- The simulation of the full pipeline to predict on user input and output to user.
    │   └── 5.1-interactive_platform.ipynb                        <- The mock-up version of the interactive platform on collecting the user input, predicting
    │                                                                possible classification labels of ICD-9-CM, and forecast potential diagnostic/screening
    │                                                                services.
    │
    ├── outputs            <- Outputs for Classifications of Diseases and Injuries, including wordclouds, error bars
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as json, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── data           <- Scripts to generate data
    │   │   └── make_dataset.py                                   <- Load and parse the ASCII OPD data files with SPSS code, save to ./data/cleaned as CSV files
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── build_features.py                                 <- The main scripts to prepare and process the datasets, incl. handling missing values, convert │   │   │                                                        and combining text descriptions, computing TF-IDF, extracting topic features and export
    │   │   │                                                        corresponding figures.
    │   │   └── combine_textual.py                                <- The scripts used for combining textual features for the clustering apporoach.
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── dvc.yaml           <- The scripts to run the pipeline and track the results


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
