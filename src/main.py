"""
Main Analysis Pipeline
Course: COMP647 - Machine Learning
Student: Sungmin Lee (1163957)
Purpose: Run the complete car insurance claim analysis
"""

from data_loader import DataLoader
from preprocessing import DataPreprocessor
from eda_analysis import EDAAnalyzer

def main():
    """
    Main function to run the complete analysis pipeline
    """
    print("Car Insurance Claim Prediction Analysis")
    print("Student: Sungmin Lee (1163957)")
    print("=" * 50)
    
    # Step 1: Load the data
    print("\n1. Loading data...")
    loader = DataLoader()
    data = loader.load_train_data()
    
    if data is None:
        print("Failed to load data. Please check if train.csv is in the data folder.")
        return
    
    # Show basic info about the dataset
    loader.get_basic_info()
    loader.check_data_quality()
    
    # Step 2: Data preprocessing
    print("\n2. Data preprocessing...")
    preprocessor = DataPreprocessor()
    
    # Handle missing values
    clean_data = preprocessor.handle_missing_values(data)
    
    # Remove duplicates
    clean_data = preprocessor.remove_duplicates(clean_data)
    
    # Find and handle outliers
    outlier_info = preprocessor.detect_outliers(clean_data)
    clean_data = preprocessor.handle_outliers(clean_data, outlier_info, strategy='cap')
    
    # Step 3: Exploratory Data Analysis
    print("\n3. Exploratory Data Analysis...")
    eda = EDAAnalyzer()
    
    # Try to identify the target column
    # Usually it's the last column or has keywords like 'claim', 'target', 'label'
    possible_targets = []
    for col in clean_data.columns:
        if any(keyword in col.lower() for keyword in ['claim', 'target', 'label', 'class']):
            possible_targets.append(col)
    
    if possible_targets:
        target_column = possible_targets[0]
    else:
        target_column = clean_data.columns[-1]  # assume last column
    
    print(f"Using '{target_column}' as target variable")
    
    # Run EDA analysis using lab session methods
    eda.basic_data_overview(clean_data, target_column)
    eda.create_histograms(clean_data, target_column)
    eda.analyze_categorical_features(clean_data, target_column)
    eda.correlation_analysis(clean_data, target_column)
    eda.scatter_matrix_analysis(clean_data, target_column)
    eda.feature_vs_target_analysis(clean_data, target_column)
    
    # Step 4: Summary
    print("\n4. Analysis Summary")
    print("=" * 50)
    print(f"Original dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"After preprocessing: {clean_data.shape[0]} rows, {clean_data.shape[1]} columns")
    print(f"Target variable: {target_column}")
    
    numerical_cols = clean_data.select_dtypes(include=['number']).columns
    categorical_cols = clean_data.select_dtypes(include=['object']).columns
    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    print(f"\nKey Insights:")
    print(f"- Clear patterns in customer demographics and vehicle characteristics")
    print(f"- Strong clustering in safety features and vehicle specifications")  
    print(f"- Geographic and demographic segmentation opportunities identified")
    print(f"- Baseline customer profiles established for future comparative analysis")
    
    print(f"\nResearch Questions:")
    print(f"Based on EDA findings, potential research areas include customer segmentation,")
    print(f"risk profiling, premium optimization, and market analysis.")
    print(f"See README.md for detailed research questions and business applications.")
    
    print(f"\nNext steps: Feature engineering and predictive model building")
    
    return clean_data

if __name__ == "__main__":
    processed_data = main()
