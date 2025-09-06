# Car Insurance Claim Prediction
COMP647 Machine Learning Assignment  
Student: Sungmin Lee (1163957)

## Project Background
As a financial supporter at Allied Financial in Christchurch, New Zealand, I handle car insurance claims daily. When customers experience accidents, they call me to help process their claims - I prepare documentation and submit it to insurance companies on their behalf, then follow up throughout the process.

This project applies machine learning to real business challenges I face: predicting which clients are likely to file claims and improving our proactive customer service approach.

## Project Overview
Predict whether a car insurance policyholder will file a claim using machine learning classification models, with insights from real-world insurance claim processing experience.

## Dataset
**Source**: Kaggle - Car Insurance Claim Prediction  
**URL**: https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification  
**Files**: train.csv, test.csv, sample_submission.csv

## Project Files
- `src/main.py` - Main analysis pipeline
- `src/data_loader.py` - Data loading and basic inspection
- `src/preprocessing.py` - Data preprocessing (missing values, outliers, encoding)
- `src/eda_analysis.py` - Exploratory data analysis
- `data/` - Dataset folder (add train.csv here)

## How to Run
1. Download dataset from Kaggle and place train.csv in `data/` folder
2. Install dependencies: `pip install -r requirements.txt`
3. Run analysis: `python src/main.py`

## Analysis Steps
1. **Data Loading** - Load and inspect the insurance dataset
2. **Data Preprocessing** - Handle missing values, outliers, and encoding
3. **Exploratory Data Analysis** - Analyze relationships between features and target
4. **Feature Analysis** - Identify important predictors for insurance claims

## Author
**Sungmin Lee**  
Student Registration Number: 1163957  
Course: COMP647 - Machine Learning  
GitHub Repository: https://github.com/Alex-Lee-1163957/COMP647Project_SungminLee
