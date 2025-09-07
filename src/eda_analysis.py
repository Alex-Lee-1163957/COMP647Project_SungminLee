"""
Exploratory Data Analysis (EDA) Module
Course: COMP647 - Machine Learning
Student: Sungmin Lee (1163957)
Purpose: Investigate correlations and relationships among features using lab session methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

class EDAAnalyzer:
    """
    Class to perform exploratory data analysis using methods from lab sessions
    """
    
    def __init__(self, figsize=(10, 6)):
        """
        Initialize EDA Analyzer with basic matplotlib settings
        """
        self.figsize = figsize
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 10
        
    def basic_data_overview(self, data, target_column):
        """
        Get basic overview of the dataset using pandas methods from lab sessions
        
        This follows the standard approach shown in class:
        1. Check data structure with info()
        2. Get statistical summary with describe() 
        3. Look at target variable distribution
        """
        print("="*60)
        print("BASIC DATA OVERVIEW")
        print("="*60)
        
        # Basic dataset information - this is always the first step in EDA
        print(f"Dataset shape: {data.shape}")
        print(f"Number of rows: {data.shape[0]:,}")
        print(f"Number of columns: {data.shape[1]}")
        
        print(f"\nData types and missing values:")
        print(data.info())
        
        print(f"\nBasic statistical summary:")
        print(data.describe())
        
        # Target variable analysis
        if target_column in data.columns:
            print(f"\nTarget variable '{target_column}' distribution:")
            target_counts = data[target_column].value_counts()
            print(target_counts)
            
            # Calculate percentages like we learned in class
            target_percentages = data[target_column].value_counts(normalize=True) * 100
            print(f"\nTarget variable percentages:")
            for value, percentage in target_percentages.items():
                print(f"  {value}: {percentage:.1f}%")
    
    def create_histograms(self, data, target_column=None):
        """
        Create histograms for all numerical features
        
        This is one of the first visualization techniques we learned in lab sessions.
        Histograms help us understand the distribution of each feature.
        """
        print(f"\nCreating histograms for numerical features...")
        
        # Get numerical columns only
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if target_column and target_column in numerical_cols:
            numerical_cols = numerical_cols.drop(target_column)
        
        if len(numerical_cols) == 0:
            print("No numerical features found for histograms.")
            return
        
        # Create histograms using pandas hist() method as shown in class
        # This creates a subplot for each numerical column automatically
        data[numerical_cols].hist(bins=30, figsize=(15, 10))
        plt.suptitle('Distribution of Numerical Features', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        print(f"Created histograms for {len(numerical_cols)} numerical features")
    
    def analyze_categorical_features(self, data, target_column=None):
        """
        Analyze categorical features using value_counts() and bar plots
        
        This follows the approach from lab sessions for categorical data exploration
        """
        print(f"\nAnalyzing categorical features...")
        
        # Get categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        if target_column and target_column in categorical_cols:
            categorical_cols = categorical_cols.drop(target_column)
        
        if len(categorical_cols) == 0:
            print("No categorical features found.")
            return
        
        print(f"Found {len(categorical_cols)} categorical features: {list(categorical_cols)}")
        
        # Analyze each categorical feature
        for col in categorical_cols:
            print(f"\n--- {col} ---")
            
            # Use value_counts() as taught in class
            value_counts = data[col].value_counts()
            print(f"Number of unique values: {data[col].nunique()}")
            print("Value distribution:")
            
            # Show top 10 values to avoid cluttering
            top_values = value_counts.head(10)
            for value, count in top_values.items():
                percentage = (count / len(data)) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
            
            if len(value_counts) > 10:
                print(f"  ... and {len(value_counts) - 10} more categories")
        
        # Create bar plots for categorical features
        n_categorical = len(categorical_cols)
        if n_categorical > 0:
            # Calculate subplot dimensions
            n_cols = min(2, n_categorical)
            n_rows = (n_categorical + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(categorical_cols):
                # Show only top 10 categories to keep plots readable
                top_categories = data[col].value_counts().head(10)
                
                # Create bar plot using pandas plot.bar() as shown in class
                top_categories.plot.bar(ax=axes[i], rot=45)
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_ylabel('Count')
            
            # Remove empty subplots
            for i in range(n_categorical, len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.show()
    
    def correlation_analysis(self, data, target_column=None):
        """
        Perform correlation analysis using pandas corr() method from lab sessions
        
        This is a key technique we learned for understanding relationships between features
        """
        print(f"\nPerforming correlation analysis...")
        
        # Get numerical columns for correlation analysis
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            print("Need at least 2 numerical features for correlation analysis.")
            return
        
        # Calculate correlation matrix using pandas corr() as taught in class
        corr_matrix = data[numerical_cols].corr()
        
        print(f"Correlation matrix for {len(numerical_cols)} numerical features:")
        print(corr_matrix.round(3))
        
        # If we have a target variable, show correlations with target
        if target_column and target_column in numerical_cols:
            print(f"\nCorrelations with target variable '{target_column}':")
            target_correlations = corr_matrix[target_column].drop(target_column)
            
            # Sort by absolute correlation value to see strongest relationships
            target_correlations_sorted = target_correlations.abs().sort_values(ascending=False)
            
            for feature in target_correlations_sorted.index:
                corr_value = target_correlations[feature]
                print(f"  {feature}: {corr_value:.3f}")
        
        # Create correlation heatmap using matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Correlation Coefficient')
        
        # Add feature names to axes
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.title('Correlation Matrix Heatmap')
        
        # Add correlation values to the plot
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='black', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def scatter_matrix_analysis(self, data, target_column=None):
        """
        Create scatter matrix using pandas scatter_matrix as shown in lab sessions
        
        This is an important visualization technique we learned for seeing 
        relationships between multiple numerical features at once
        """
        print(f"\nCreating scatter matrix...")
        
        # Get numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        # Select most important features for scatter matrix (to keep it readable)
        # Include target column if it's numerical
        if target_column and target_column in numerical_cols:
            # Put target column first, then select other important features
            features_for_scatter = [target_column]
            other_features = [col for col in numerical_cols if col != target_column]
            
            # Add up to 4 more features to keep the plot manageable
            features_for_scatter.extend(other_features[:4])
        else:
            # Just take first 5 numerical features
            features_for_scatter = numerical_cols[:5].tolist()
        
        if len(features_for_scatter) < 2:
            print("Need at least 2 numerical features for scatter matrix.")
            return
        
        print(f"Creating scatter matrix for features: {features_for_scatter}")
        
        # Create scatter matrix using pandas function as taught in class
        scatter_matrix(data[features_for_scatter], figsize=(12, 10), alpha=0.6)
        plt.suptitle('Scatter Matrix of Key Features', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def feature_vs_target_analysis(self, data, target_column):
        """
        Analyze how each feature relates to the target variable
        
        This helps us understand which features might be most important for prediction
        """
        if target_column not in data.columns:
            print(f"Target column '{target_column}' not found in dataset.")
            return
        
        print(f"\nAnalyzing features vs target variable '{target_column}'...")
        
        # For numerical target, we can look at correlations
        if data[target_column].dtype in ['int64', 'float64']:
            print("Target is numerical - analyzing correlations:")
            
            numerical_features = data.select_dtypes(include=[np.number]).columns
            numerical_features = numerical_features.drop(target_column)
            
            correlations = []
            for feature in numerical_features:
                corr = data[feature].corr(data[target_column])
                correlations.append((feature, corr))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print("Features ranked by correlation with target:")
            for feature, corr in correlations:
                print(f"  {feature}: {corr:.3f}")
        
        # For categorical target, show distribution differences
        else:
            print("Target is categorical - analyzing distributions:")
            
            # Get unique target values
            target_values = data[target_column].unique()
            print(f"Target has {len(target_values)} categories: {target_values}")
            
            # For each numerical feature, show basic stats by target category
            numerical_features = data.select_dtypes(include=[np.number]).columns
            
            for feature in numerical_features[:3]:  # Show first 3 to avoid too much output
                print(f"\n{feature} by {target_column}:")
                feature_by_target = data.groupby(target_column)[feature].agg(['mean', 'std', 'count'])
                print(feature_by_target.round(2))


def main():
    """
    Main function to demonstrate EDA analysis using lab session methods
    """
    from data_loader import DataLoader
    from preprocessing import DataPreprocessor
    
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS - LAB SESSION METHODS")
    print("Student: Sungmin Lee (1163957)")
    print("="*60)
    
    # Load and preprocess data
    loader = DataLoader()
    data = loader.load_train_data()
    
    if data is None:
        print("Could not load data. Please check data_loader.py")
        return
    
    # Basic preprocessing
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.handle_missing_values(data)
    clean_data = preprocessor.remove_duplicates(clean_data)
    
    # Initialize EDA analyzer
    eda = EDAAnalyzer()
    
    # Try to identify target column automatically
    # Look for common target column names in insurance datasets
    possible_targets = []
    for col in clean_data.columns:
        if any(keyword in col.lower() for keyword in ['claim', 'target', 'label', 'outcome', 'class']):
            possible_targets.append(col)
    
    if possible_targets:
        target_column = possible_targets[0]
    else:
        target_column = clean_data.columns[-1]  # assume last column
    
    print(f"Using '{target_column}' as target variable")
    
    # Run EDA analysis following lab session structure
    
    # 1. Basic data overview
    eda.basic_data_overview(clean_data, target_column)
    
    # 2. Histograms for numerical features
    eda.create_histograms(clean_data, target_column)
    
    # 3. Categorical feature analysis
    eda.analyze_categorical_features(clean_data, target_column)
    
    # 4. Correlation analysis
    eda.correlation_analysis(clean_data, target_column)
    
    # 5. Scatter matrix
    eda.scatter_matrix_analysis(clean_data, target_column)
    
    # 6. Feature vs target analysis
    eda.feature_vs_target_analysis(clean_data, target_column)
    
    print(f"\n" + "="*60)
    print("EDA ANALYSIS COMPLETED")
    print("="*60)
    print("Key findings from this analysis can help us understand:")
    print("1. Which features have the strongest relationships with the target")
    print("2. How features are distributed and if they need transformation")
    print("3. Whether there are any obvious patterns in the data")
    print("4. Which features might be most useful for building predictive models")


if __name__ == "__main__":
    main()