"""
Exploratory Data Analysis (EDA) Module
Course: COMP647 - Machine Learning
Student: Sungmin Lee (1163957)
Purpose: Investigate correlations and potential relationships among features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class EDAAnalyzer:
    """Class to perform comprehensive exploratory data analysis"""
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialize EDA Analyzer
        Args:
            figsize (tuple): Default figure size for plots
        """
        self.figsize = figsize
        # Set visualization style for better appearance
        plt.style.use('default')
        sns.set_palette("husl")
        
    def analyze_target_variable(self, data, target_column):
        """
        Analyze the target variable distribution and characteristics
        
        EXPLANATION: Understanding the target variable is crucial for:
        - Identifying class imbalance issues
        - Choosing appropriate evaluation metrics
        - Understanding the business problem context
        
        Args:
            data (pd.DataFrame): Input dataset
            target_column (str): Name of target column
        """
        print(f"\nAnalyzing target variable: {target_column}")
        
        if target_column not in data.columns:
            print(f"Error: Column '{target_column}' not found in the dataset!")
            return
        
        target_data = data[target_column]
        
        # BASIC STATISTICS
        print(f"Target variable: {target_column}")
        print(f"Data type: {target_data.dtype}")
        print(f"Total records: {len(target_data)}")
        print(f"Missing values: {target_data.isnull().sum()}")
        
        # VALUE DISTRIBUTION
        print(f"\nValue Distribution:")
        value_counts = target_data.value_counts()
        for value, count in value_counts.items():
            percentage = (count / len(target_data)) * 100
            print(f"  {value}: {count} ({percentage:.1f}%)")
        
        # CLASS BALANCE ANALYSIS
        # Important for classification problems - imbalanced classes can bias models
        if len(value_counts) == 2:
            minority_class = value_counts.min()
            majority_class = value_counts.max()
            imbalance_ratio = majority_class / minority_class
            
            print(f"\nClass Balance Analysis:")
            print(f"Majority class: {majority_class} samples")
            print(f"Minority class: {minority_class} samples")
            print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 3:
                print("Warning: Classes are imbalanced!")
                print("This might affect model performance - consider resampling techniques")
        
        # VISUALIZATION
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Count plot
        sns.countplot(data=data, x=target_column, ax=axes[0])
        axes[0].set_title(f'Distribution of {target_column}')
        axes[0].set_xlabel(target_column)
        axes[0].set_ylabel('Count')
        
        # Pie chart
        axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        axes[1].set_title(f'Proportion of {target_column}')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_numerical_features(self, data, target_column=None):
        """
        Analyze numerical features and their relationships
        
        EXPLANATION: Numerical feature analysis helps identify:
        - Distribution patterns (normal, skewed, bimodal)
        - Potential outliers and anomalies  
        - Relationships with target variable
        - Features that might need transformation
        
        Args:
            data (pd.DataFrame): Input dataset
            target_column (str): Target variable for relationship analysis
        """
        print(f"\nAnalyzing numerical features")
        
        # Get numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        if len(numerical_cols) == 0:
            print("No numerical features found in the dataset.")
            return
        
        print(f"Analyzing {len(numerical_cols)} numerical features:")
        print(f"Features: {numerical_cols}")
        
        # DESCRIPTIVE STATISTICS
        print(f"\nDescriptive Statistics:")
        print(data[numerical_cols].describe())
        
        # DISTRIBUTION ANALYSIS
        # Create distribution plots for each numerical feature
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            # HISTOGRAM WITH DENSITY CURVE
            # Shows distribution shape: normal, skewed, multimodal
            sns.histplot(data[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            
            # Add statistics text
            mean_val = data[col].mean()
            median_val = data[col].median()
            std_val = data[col].std()
            skewness = data[col].skew()
            
            # Interpretation of skewness:
            # |skew| < 0.5: approximately symmetric
            # 0.5 < |skew| < 1: moderately skewed
            # |skew| > 1: highly skewed
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.2f}')
            axes[i].legend()
            
            print(f"\n{col} Statistics:")
            print(f"  Mean: {mean_val:.2f}, Median: {median_val:.2f}")
            print(f"  Std: {std_val:.2f}, Skewness: {skewness:.2f}")
            
            # Skewness interpretation
            if abs(skewness) < 0.5:
                skew_desc = "approximately symmetric"
            elif abs(skewness) < 1:
                skew_desc = "moderately skewed"
            else:
                skew_desc = "highly skewed"
            print(f"  Distribution: {skew_desc}")
        
        # Remove empty subplots
        for i in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
        
        # CORRELATION WITH TARGET (if available)
        if target_column and target_column in data.columns:
            self._analyze_numerical_target_relationships(data, numerical_cols, target_column)
    
    def _analyze_numerical_target_relationships(self, data, numerical_cols, target_column):
        """
        Analyze relationships between numerical features and target variable
        
        Args:
            data (pd.DataFrame): Input dataset
            numerical_cols (list): List of numerical column names
            target_column (str): Target variable name
        """
        print(f"\n=== NUMERICAL FEATURES vs TARGET ANALYSIS ===")
        
        # Calculate correlations with target variable
        correlations = []
        for col in numerical_cols:
            try:
                # Use appropriate correlation method based on target type
                if data[target_column].dtype in ['object', 'category']:
                    # For categorical target, use point-biserial correlation
                    # Convert target to numerical for correlation calculation
                    target_numeric = pd.Categorical(data[target_column]).codes
                    corr, p_value = pearsonr(data[col].fillna(0), target_numeric)
                else:
                    # For numerical target, use Pearson correlation
                    corr, p_value = pearsonr(data[col].fillna(0), data[target_column].fillna(0))
                
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'p_value': p_value,
                    'significance': 'Significant' if p_value < 0.05 else 'Not Significant'
                })
            except:
                correlations.append({
                    'feature': col,
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'significance': 'Error'
                })
        
        # Display correlation results
        corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
        print("\nCorrelation with Target Variable:")
        print(corr_df.to_string(index=False))
        
        # INTERPRETATION GUIDE
        print(f"\nCorrelation Interpretation Guide:")
        print(f"  |r| > 0.7: Strong relationship")
        print(f"  0.3 < |r| < 0.7: Moderate relationship") 
        print(f"  |r| < 0.3: Weak relationship")
        print(f"  p < 0.05: Statistically significant")
        
        # BOX PLOTS for categorical target
        if data[target_column].dtype in ['object', 'category']:
            n_cols = min(3, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numerical_cols):
                # Box plot shows distribution differences between target classes
                sns.boxplot(data=data, x=target_column, y=col, ax=axes[i])
                axes[i].set_title(f'{col} by {target_column}')
                axes[i].tick_params(axis='x', rotation=45)
            
            # Remove empty subplots
            for i in range(len(numerical_cols), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.show()
    
    def analyze_categorical_features(self, data, target_column=None):
        """
        Analyze categorical features and their relationships
        
        EXPLANATION: Categorical feature analysis reveals:
        - Category distributions and frequencies
        - Relationships between categorical variables
        - Association with target variable (Chi-square tests)
        - Potential categories for grouping/encoding
        
        Args:
            data (pd.DataFrame): Input dataset
            target_column (str): Target variable for relationship analysis
        """
        print(f"\n=== CATEGORICAL FEATURES ANALYSIS ===")
        
        # Get categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column and target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        if len(categorical_cols) == 0:
            print("No categorical features found!")
            return
        
        print(f"Analyzing {len(categorical_cols)} categorical features:")
        print(f"Features: {categorical_cols}")
        
        # CATEGORY DISTRIBUTION ANALYSIS
        for col in categorical_cols:
            print(f"\n--- {col} Analysis ---")
            
            # Basic statistics
            unique_count = data[col].nunique()
            total_count = len(data[col])
            missing_count = data[col].isnull().sum()
            
            print(f"Unique categories: {unique_count}")
            print(f"Missing values: {missing_count}")
            
            # Value counts and percentages
            value_counts = data[col].value_counts()
            print(f"Category distribution:")
            for category, count in value_counts.head(10).items():  # Show top 10
                percentage = (count / total_count) * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")
            
            if len(value_counts) > 10:
                print(f"  ... and {len(value_counts) - 10} more categories")
            
            # HIGH CARDINALITY WARNING
            # High cardinality can cause issues with encoding and model performance
            if unique_count > 20:
                print(f"  ⚠️  HIGH CARDINALITY DETECTED ({unique_count} categories)")
                print(f"  Consider: grouping rare categories, target encoding, or dimensionality reduction")
        
        # VISUALIZATION
        # Create bar plots for categorical features
        n_cols = min(2, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            # Show only top categories to avoid cluttered plots
            top_categories = data[col].value_counts().head(15)
            
            sns.barplot(x=top_categories.values, y=top_categories.index, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel('Count')
            axes[i].set_ylabel(col)
        
        # Remove empty subplots
        for i in range(len(categorical_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
        
        # CATEGORICAL-TARGET RELATIONSHIPS
        if target_column and target_column in data.columns:
            self._analyze_categorical_target_relationships(data, categorical_cols, target_column)
    
    def _analyze_categorical_target_relationships(self, data, categorical_cols, target_column):
        """
        Analyze relationships between categorical features and target variable
        
        Args:
            data (pd.DataFrame): Input dataset  
            categorical_cols (list): List of categorical column names
            target_column (str): Target variable name
        """
        print(f"\n=== CATEGORICAL FEATURES vs TARGET ANALYSIS ===")
        
        chi_square_results = []
        
        for col in categorical_cols:
            try:
                # CHI-SQUARE TEST OF INDEPENDENCE
                # Tests whether two categorical variables are independent
                # H0: Variables are independent (no relationship)
                # H1: Variables are dependent (relationship exists)
                
                # Create contingency table
                contingency_table = pd.crosstab(data[col], data[target_column])
                
                # Perform chi-square test
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                # Calculate effect size (Cramér's V)
                # Cramér's V ranges from 0 (no association) to 1 (perfect association)
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                
                chi_square_results.append({
                    'feature': col,
                    'chi_square': chi2,
                    'p_value': p_value,
                    'cramers_v': cramers_v,
                    'significance': 'Significant' if p_value < 0.05 else 'Not Significant',
                    'effect_size': 'Large' if cramers_v > 0.5 else 'Medium' if cramers_v > 0.3 else 'Small'
                })
                
                print(f"\n{col} vs {target_column}:")
                print(f"  Chi-square: {chi2:.4f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  Cramér's V: {cramers_v:.4f}")
                print(f"  Relationship: {chi_square_results[-1]['significance']} ({chi_square_results[-1]['effect_size']} effect)")
                
            except Exception as e:
                print(f"\n{col} vs {target_column}: Error in analysis ({str(e)})")
                chi_square_results.append({
                    'feature': col,
                    'chi_square': np.nan,
                    'p_value': np.nan,
                    'cramers_v': np.nan,
                    'significance': 'Error',
                    'effect_size': 'Error'
                })
        
        # Summary table
        if chi_square_results:
            chi_df = pd.DataFrame(chi_square_results).sort_values('cramers_v', ascending=False)
            print(f"\nCategorical Features Relationship Summary:")
            print(chi_df.to_string(index=False))
            
            print(f"\nInterpretation Guide:")
            print(f"  p < 0.05: Statistically significant relationship")
            print(f"  Cramér's V > 0.5: Large effect size")
            print(f"  Cramér's V 0.3-0.5: Medium effect size")
            print(f"  Cramér's V < 0.3: Small effect size")
    
    def create_correlation_matrix(self, data, target_column=None):
        """
        Create and visualize correlation matrix for numerical features
        
        EXPLANATION: Correlation matrix reveals:
        - Linear relationships between numerical variables
        - Potential multicollinearity issues
        - Feature redundancy
        - Important predictive relationships
        
        Args:
            data (pd.DataFrame): Input dataset
            target_column (str): Target variable to highlight
        """
        print(f"\n=== CORRELATION MATRIX ANALYSIS ===")
        
        # Get numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            print("Need at least 2 numerical features for correlation analysis!")
            return
        
        # Calculate correlation matrix
        correlation_matrix = data[numerical_cols].corr()
        
        print(f"Correlation matrix for {len(numerical_cols)} numerical features")
        
        # VISUALIZATION
        plt.figure(figsize=(12, 10))
        
        # Create heatmap with annotations
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.show()
        
        # MULTICOLLINEARITY ANALYSIS
        # High correlations (|r| > 0.8) may indicate multicollinearity
        print(f"\nMulticollinearity Analysis:")
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_corr_pairs.append({
                        'feature_1': correlation_matrix.columns[i],
                        'feature_2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if high_corr_pairs:
            print(f"⚠️  HIGH CORRELATIONS DETECTED (|r| > 0.8):")
            for pair in high_corr_pairs:
                print(f"  {pair['feature_1']} ↔ {pair['feature_2']}: {pair['correlation']:.3f}")
            print(f"Consider: removing redundant features or using PCA")
        else:
            print(f"✅ No severe multicollinearity detected")
        
        # TARGET CORRELATIONS (if available)
        if target_column and target_column in numerical_cols:
            print(f"\nCorrelations with Target Variable ({target_column}):")
            target_corrs = correlation_matrix[target_column].drop(target_column).sort_values(key=abs, ascending=False)
            
            for feature, corr in target_corrs.items():
                strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
                print(f"  {feature}: {corr:.3f} ({strength})")


def main():
    """Main function to demonstrate EDA analysis"""
    from data_loader import DataLoader
    from preprocessing import DataPreprocessor
    
    print("=== CAR INSURANCE CLAIM - EXPLORATORY DATA ANALYSIS ===\n")
    
    # Load data
    loader = DataLoader()
    data = loader.load_train_data()
    
    if data is None:
        print("Could not load data. Please check data_loader.py")
        return
    
    # Basic preprocessing for EDA
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.handle_missing_values(data)
    clean_data = preprocessor.remove_duplicates(clean_data)
    
    # Create EDA analyzer
    eda = EDAAnalyzer()
    
    # Identify target column (usually the last column in classification datasets)
    # You may need to adjust this based on your actual dataset
    target_column = clean_data.columns[-1]  # Assuming target is last column
    print(f"Assumed target variable: {target_column}")
    print("If this is incorrect, please specify the correct target column name.")
    
    # COMPREHENSIVE EDA ANALYSIS
    
    # 1. Target Variable Analysis
    eda.analyze_target_variable(clean_data, target_column)
    
    # 2. Numerical Features Analysis
    eda.analyze_numerical_features(clean_data, target_column)
    
    # 3. Categorical Features Analysis  
    eda.analyze_categorical_features(clean_data, target_column)
    
    # 4. Correlation Matrix
    eda.create_correlation_matrix(clean_data, target_column)
    
    print("\n=== EDA ANALYSIS COMPLETED ===")
    print("\nKey Insights for Model Development:")
    print("1. Check for class imbalance in target variable")
    print("2. Identify most predictive features from correlation analysis")
    print("3. Consider feature engineering for high-cardinality categorical variables")
    print("4. Address multicollinearity if detected")
    print("5. Plan appropriate preprocessing based on feature distributions")


if __name__ == "__main__":
    main()
