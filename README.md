# Car Insurance Claim Prediction
COMP647 Machine Learning Assignment  
Student: Sungmin Lee (1163957)

## About Me & Project Background
I am currently working as a Financial Supporter at Allied Financial in Christchurch, New Zealand. In my daily work, I handle car insurance claims for customers - when they experience accidents, they call me for assistance. I help them prepare the necessary documentation, submit claims to insurance companies on their behalf, and follow up throughout the entire claims process.

This hands-on experience with insurance claims processing has given me valuable insights into the industry, which I wanted to apply to this machine learning project. The goal is to use data science techniques to predict which clients are likely to file claims, potentially helping improve proactive customer service and operational efficiency.

## Project Overview
Predict whether a car insurance policyholder will file a claim using machine learning classification models, with insights from real-world insurance claim processing experience.

## Dataset
**Source**: Kaggle - Car Insurance Claim Prediction  
**URL**: https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification  
**Files**: train.csv, test.csv, sample_submission.csv

## Project Files
- `Assignment-2.py` - Single-file submission (data loading, preprocessing, EDA, insights)
- `data/` - Dataset folder (add `train.csv` here)

## How to Run
1. Download dataset from Kaggle and place `train.csv` in `data/` folder
2. Install dependencies: `pip install -r requirements.txt`
3. Run analysis: `python Assignment-2.py`
4. Outputs: plots in `docs/plots/`, summary in `docs/insights.md`

## Analysis Steps
1. **Data Loading** - Load and inspect the insurance dataset
2. **Data Preprocessing** - Handle missing values, outliers, and encoding
3. **Exploratory Data Analysis** - Analyze relationships between features and target
4. **Feature Analysis** - Identify important predictors for insurance claims

## Research Questions

Based on the exploratory data analysis performed, several interesting research questions emerge from this car insurance dataset:

### 1. Customer Segmentation Analysis
**Question:** "Can we identify distinct customer segments based on vehicle and demographic characteristics?"

**EDA Evidence:**
- Population density shows bimodal distribution (urban vs rural customers)
- Age of policyholder follows normal distribution around middle age
- Vehicle safety features (airbags, NCAP rating) show clear clustering patterns

**Business Value:** From my experience at Allied Financial, different customer segments require different service approaches. Understanding these segments helps in providing targeted services and preparing appropriate documentation.

### 2. Risk Profiling for Proactive Service
**Question:** "What vehicle and customer characteristics indicate higher maintenance or service needs?"

**EDA Evidence:**
- Age of car is heavily skewed toward newer vehicles (most under 5% of maximum age)
- Safety features vary significantly across the dataset
- Vehicle specifications show distinct clustering patterns

**Business Application:** Understanding which customers might need more support helps in proactive service delivery and documentation preparation.

### 3. Premium Optimization Analysis
**Question:** "How do vehicle safety features correlate with customer demographics for pricing strategies?"

**EDA Evidence:**
- Clear patterns in safety features (NCAP ratings concentrated at 0, 2, and 3)
- Vehicle size specifications cluster around specific standard values
- Age and policy tenure show interesting distribution patterns

**Industry Relevance:** Understanding relationships between safety features and customer profiles helps insurance companies set appropriate premiums and identify low-risk segments.

### 4. Service Preparation and Documentation
**Question:** "Can we predict what type of documentation and service support different customers will need?"

**EDA Evidence:**
- Vehicle complexity varies significantly (different engine types, transmission systems)
- Customer tenure shows wide distribution indicating varying experience levels
- Geographic clustering suggests regional service patterns

**Real-world Application:** In daily work, customers with newer, more complex vehicles often need more detailed documentation. Understanding these patterns helps prepare appropriate paperwork templates in advance.

### 5. Market Analysis and Expansion
**Question:** "What market segments are underrepresented and could be targets for business expansion?"

**EDA Evidence:**
- Vehicle make and model distributions show market concentration
- Geographic distribution reveals potential gaps
- Age demographics concentrated in specific ranges

**Strategic Value:** Identifying underrepresented segments helps target marketing efforts and expand customer base.

## Key Findings

Although the current dataset shows `is_claim = 0` for all records, this provides valuable insights:

- **Baseline Profiling**: Establishes profiles of 'safe' customers for comparative analysis
- **Proactive Service**: Understanding safe customer patterns helps identify deviations
- **Business Intelligence**: Customer behavior patterns provide valuable business insights even without claims data
- **Market Understanding**: Clear clustering in vehicle features and demographics reveals market structure

The EDA revealed meaningful distribution patterns and feature relationships that support these research questions and provide practical value for insurance industry applications.

## Author
**Sungmin Lee**  
Student Registration Number: 1163957  
Course: COMP647 - Machine Learning  
GitHub Repository: https://github.com/Alex-Lee-1163957/COMP647Project_SungminLee
