"""
Data Preprocessing Module
Course: COMP647 - Machine Learning
Student: Sungmin Lee (1163957)
Purpose: Clean and preprocess the car insurance claim dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocessor:
    """Class to handle data preprocessing tasks"""
    
    def __init__(self):
        """Initialize DataPreprocessor"""
        pass
        
    def handle_missing_values(self, data, strategy='auto'):
        """
        Handle missing values in the dataset using appropriate imputation methods
        
        EXPLANATION: Missing values are common in real-world datasets and can significantly
        impact model performance. We use different strategies based on data characteristics:
        - For categorical data: Mode imputation (most frequent value)
        - For numerical data: Median imputation (robust to outliers)
        
        Args:
            data (pd.DataFrame): Input dataset
            strategy (str): Strategy for handling missing values
                          'auto' - automatic based on data type (RECOMMENDED)
                          'drop' - drop rows with missing values
                          'fill' - fill with mean/mode
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        print(f"\nHandling missing values using {strategy} strategy")
        
        # First, let's see how many missing values we have
        initial_missing = data.isnull().sum().sum()
        print(f"Total missing values: {initial_missing}")
        
        if initial_missing == 0:
            print("Great! No missing values found.")
            return data.copy()
        
        processed_data = data.copy()
        
        if strategy == 'drop':
            # Simple approach: just remove rows with missing data
            # This works well when we don't have too many missing values
            processed_data = processed_data.dropna()
            print(f"Rows remaining after dropping: {len(processed_data)}")
            
        elif strategy == 'fill' or strategy == 'auto':
            # Fill missing values with reasonable estimates
            
            for col in processed_data.columns:
                if processed_data[col].isnull().sum() > 0:
                    if processed_data[col].dtype in ['object']:
                        # For text/category columns, use the most common value
                        # In my experience at Allied Financial, missing categorical data 
                        # (like vehicle type or education) often follows the same pattern 
                        # as our typical customer base
                        mode_value = processed_data[col].mode()[0]
                        processed_data[col].fillna(mode_value, inplace=True)
                        print(f"  {col}: filled with most common value '{mode_value}'")
                    else:
                        # For number columns, use median (middle value)
                        # From processing hundreds of claims, I've seen that income and age data
                        # can have extreme outliers. Median is more representative of typical customers
                        # who call in for claim assistance
                        median_value = processed_data[col].median()
                        processed_data[col].fillna(median_value, inplace=True)
                        print(f"  {col}: filled with median value {median_value}")
        
        # Check if we successfully handled all missing values
        final_missing = processed_data.isnull().sum().sum()
        print(f"Missing values after processing: {final_missing}")
        
        return processed_data
    
    def remove_duplicates(self, data):
        """
        Remove duplicate rows from the dataset
        Args:
            data (pd.DataFrame): Input dataset
        Returns:
            pd.DataFrame: Dataset with duplicates removed
        """
        print(f"\n=== REMOVING DUPLICATES ===")
        
        initial_rows = len(data)
        duplicates = data.duplicated().sum()
        
        print(f"Initial rows: {initial_rows}")
        print(f"Duplicate rows found: {duplicates}")
        
        if duplicates > 0:
            processed_data = data.drop_duplicates()
            final_rows = len(processed_data)
            print(f"Rows after removing duplicates: {final_rows}")
            print(f"Rows removed: {initial_rows - final_rows}")
            return processed_data
        else:
            print("No duplicate rows found!")
            return data.copy()
    
    def detect_outliers(self, data, method='iqr'):
        """
        Detect outliers in numerical columns using statistical methods
        
        EXPLANATION: Outliers can significantly impact machine learning models,
        especially linear models. In insurance data, outliers might represent:
        - Data entry errors (impossible values)
        - Genuine extreme cases (very high-value claims)
        - Fraudulent activities (unusual patterns)
        
        Args:
            data (pd.DataFrame): Input dataset
            method (str): Method for outlier detection
                         'iqr' - Interquartile Range method (RECOMMENDED)
                         'zscore' - Z-score method
        Returns:
            dict: Dictionary with outlier information for each column
        """
        print(f"\nLooking for outliers using {method} method")
        
        # Only check numerical columns for outliers
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numerical_cols:
            if method == 'iqr':
                # IQR method: based on the box plot rule
                # Find the 25th and 75th percentiles
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Anything below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is an outlier
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count how many outliers we have
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                outlier_count = len(outliers)
                
            elif method == 'zscore':
                # Z-score method: based on standard deviations
                # If a value is more than 3 standard deviations away, it's an outlier
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = data[z_scores > 3]
                outlier_count = len(outliers)
                lower_bound = data[col].mean() - 3 * data[col].std()
                upper_bound = data[col].mean() + 3 * data[col].std()
            
            outlier_percentage = (outlier_count / len(data)) * 100
            
            # Save this info for later use
            outlier_info[col] = {
                'count': outlier_count,
                'percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            print(f"  {col}: found {outlier_count} outliers ({outlier_percentage:.1f}%)")
        
        return outlier_info
    
    def handle_outliers(self, data, outlier_info, strategy='cap'):
        """
        Handle outliers in the dataset
        Args:
            data (pd.DataFrame): Input dataset
            outlier_info (dict): Outlier information from detect_outliers
            strategy (str): Strategy for handling outliers ('cap', 'remove', 'keep')
        Returns:
            pd.DataFrame: Dataset with outliers handled
        """
        print(f"\n=== HANDLING OUTLIERS (Strategy: {strategy}) ===")
        
        processed_data = data.copy()
        
        if strategy == 'keep':
            print("Keeping all outliers as is.")
            return processed_data
        
        for col, info in outlier_info.items():
            if info['count'] > 0:
                if strategy == 'cap':
                    # Cap outliers to bounds
                    processed_data[col] = processed_data[col].clip(
                        lower=info['lower_bound'], 
                        upper=info['upper_bound']
                    )
                    print(f"  {col}: capped {info['count']} outliers")
                    
                elif strategy == 'remove':
                    # Remove outlier rows
                    mask = (processed_data[col] >= info['lower_bound']) & \
                           (processed_data[col] <= info['upper_bound'])
                    processed_data = processed_data[mask]
                    print(f"  {col}: removed rows with outliers")
        
        if strategy == 'remove':
            print(f"Final dataset shape after outlier removal: {processed_data.shape}")
        
        return processed_data
    
    def encode_categorical_variables(self, data, target_column=None):
        """
        Convert text categories to numbers using simple mapping
        
        This uses basic pandas methods to convert categorical data to numerical.
        For each text column, we create a simple mapping like:
        'Category A' -> 1, 'Category B' -> 2, etc.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_column (str): Name of target column to exclude from encoding
        Returns:
            pd.DataFrame: Dataset with encoded categorical variables
        """
        print(f"\n=== CONVERTING TEXT TO NUMBERS ===")
        
        processed_data = data.copy()
        
        # Find text columns (object type)
        text_cols = processed_data.select_dtypes(include=['object']).columns
        
        # Don't change the target column
        if target_column and target_column in text_cols:
            text_cols = text_cols.drop(target_column)
        
        print(f"Text columns to convert: {list(text_cols)}")
        
        for col in text_cols:
            unique_values = processed_data[col].nunique()
            print(f"  {col}: {unique_values} unique values")
            
            # Create simple number mapping for each category
            # Get unique categories and assign numbers 1, 2, 3, etc.
            categories = processed_data[col].unique()
            category_map = {}
            for i, category in enumerate(categories):
                category_map[category] = i + 1  # Start from 1, not 0
            
            # Apply the mapping to convert text to numbers
            processed_data[col] = processed_data[col].map(category_map)
            print(f"    Converted to numbers: {category_map}")
        
        return processed_data
    
    def normalize_features(self, data, target_column=None):
        """
        Normalize numerical features using basic math
        
        This uses simple standardization: (value - mean) / standard_deviation
        This makes all numerical features have mean=0 and std=1
        
        Args:
            data (pd.DataFrame): Input dataset
            target_column (str): Name of target column to exclude from scaling
        Returns:
            pd.DataFrame: Dataset with normalized features
        """
        print(f"\n=== NORMALIZING NUMERICAL FEATURES ===")
        
        processed_data = data.copy()
        
        # Get numerical columns (excluding target)
        numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
        if target_column and target_column in numerical_cols:
            numerical_cols = numerical_cols.drop(target_column)
        
        print(f"Numerical columns to normalize: {list(numerical_cols)}")
        
        if len(numerical_cols) > 0:
            for col in numerical_cols:
                # Simple standardization: (x - mean) / std
                mean_val = processed_data[col].mean()
                std_val = processed_data[col].std()
                
                if std_val > 0:  # Avoid division by zero
                    processed_data[col] = (processed_data[col] - mean_val) / std_val
                    print(f"  {col}: normalized (mean={mean_val:.2f}, std={std_val:.2f})")
                else:
                    print(f"  {col}: skipped (std=0)")
        else:
            print("No numerical columns found to normalize")
        
        return processed_data
    
    def get_preprocessing_summary(self, original_data, processed_data):
        """
        Generate a summary of preprocessing steps
        Args:
            original_data (pd.DataFrame): Original dataset
            processed_data (pd.DataFrame): Processed dataset
        """
        print(f"\n=== PREPROCESSING SUMMARY ===")
        print(f"Original shape: {original_data.shape}")
        print(f"Final shape: {processed_data.shape}")
        print(f"Rows changed: {original_data.shape[0] - processed_data.shape[0]}")
        print(f"Columns changed: {processed_data.shape[1] - original_data.shape[1]}")
        
        # Data types comparison
        print(f"\nData types summary:")
        original_types = original_data.dtypes.value_counts()
        final_types = processed_data.dtypes.value_counts()
        
        print("Original:")
        for dtype, count in original_types.items():
            print(f"  {dtype}: {count}")
        
        print("Final:")
        for dtype, count in final_types.items():
            print(f"  {dtype}: {count}")


def main():
    """Main function to demonstrate preprocessing"""
    from data_loader import DataLoader
    
    print("=== CAR INSURANCE CLAIM DATA PREPROCESSING ===\n")
    
    # Load data
    loader = DataLoader()
    data = loader.load_train_data()
    
    if data is None:
        print("Could not load data. Please check data_loader.py")
        return
    
    # Create preprocessor
    preprocessor = DataPreprocessor()
    
    print(f"\nOriginal data shape: {data.shape}")
    
    # Preprocessing pipeline
    processed_data = data.copy()
    
    # 1. Handle missing values
    processed_data = preprocessor.handle_missing_values(processed_data)
    
    # 2. Remove duplicates
    processed_data = preprocessor.remove_duplicates(processed_data)
    
    # 3. Detect and handle outliers
    outlier_info = preprocessor.detect_outliers(processed_data)
    processed_data = preprocessor.handle_outliers(processed_data, outlier_info, strategy='cap')
    
    # 4. Encode categorical variables (assuming target is last column)
    target_col = processed_data.columns[-1] if len(processed_data.columns) > 0 else None
    processed_data = preprocessor.encode_categorical_variables(processed_data, target_col)
    
    # 5. Normalize features
    processed_data = preprocessor.normalize_features(processed_data, target_col)
    
    # Generate summary
    preprocessor.get_preprocessing_summary(data, processed_data)
    
    print("\n=== PREPROCESSING COMPLETED ===")
    print("Processed data is ready for analysis and modeling!")
    
    return processed_data


if __name__ == "__main__":
    main()
