"""
Data Loader Module
Course: COMP647 - Machine Learning
Student: Sungmin Lee (1163957)
Purpose: Load and inspect the car insurance claim dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

class DataLoader:
    """Class to handle data loading and basic inspection"""
    
    def __init__(self, data_dir="../data"):
        """
        Initialize DataLoader
        Args:
            data_dir (str): Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.test_data = None
        
    def list_data_files(self):
        """List all files in the data directory"""
        print("Files in data directory:")
        if self.data_dir.exists():
            files = list(self.data_dir.iterdir())
            for file in files:
                print(f"  - {file.name} ({file.stat().st_size / 1024:.1f} KB)")
            return files
        else:
            print("  Data directory not found!")
            return []
    
    def load_train_data(self, filename="train.csv"):
        """
        Load training dataset
        Args:
            filename (str): Name of the training data file
        Returns:
            pd.DataFrame: Loaded training data
        """
        file_path = self.data_dir / filename
        
        try:
            if file_path.exists():
                self.train_data = pd.read_csv(file_path)
                print(f"Training data loaded successfully from: {filename}")
                print(f"Shape: {self.train_data.shape}")
                return self.train_data
            else:
                print(f" File not found: {filename}")
                return None
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def load_test_data(self, filename="test.csv"):
        """
        Load test dataset
        Args:
            filename (str): Name of the test data file
        Returns:
            pd.DataFrame: Loaded test data
        """
        file_path = self.data_dir / filename
        
        try:
            if file_path.exists():
                self.test_data = pd.read_csv(file_path)
                print(f"Test data loaded successfully from: {filename}")
                print(f"   Shape: {self.test_data.shape}")
                return self.test_data
            else:
                print(f"File not found: {filename}")
                return None
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def get_basic_info(self, data=None):
        """
        Display basic information about the dataset
        Args:
            data (pd.DataFrame): Dataset to analyze (uses train_data if None)
        """
        if data is None:
            data = self.train_data
            
        if data is None:
            print("No data available. Please load data first.")
            return
        
        print("\n=== DATASET BASIC INFORMATION ===")
        print(f"Dataset shape: {data.shape}")
        print(f"Number of rows: {data.shape[0]:,}")
        print(f"Number of columns: {data.shape[1]}")
        
        print("\n=== COLUMN INFORMATION ===")
        print("Column names and data types:")
        for i, (col, dtype) in enumerate(zip(data.columns, data.dtypes)):
            print(f"{i+1:2d}. {col:<25} - {dtype}")
        
        print("\n=== FIRST 5 ROWS ===")
        print(data.head())
        
        print("\n=== LAST 5 ROWS ===")
        print(data.tail())
    
    def check_data_quality(self, data=None):
        """
        Check data quality (missing values, duplicates, etc.)
        Args:
            data (pd.DataFrame): Dataset to check (uses train_data if None)
        """
        if data is None:
            data = self.train_data
            
        if data is None:
            print("No data available. Please load data first.")
            return
        
        print("\n=== DATA QUALITY ASSESSMENT ===")
        
        # Missing values
        print("\n1. Missing Values Analysis:")
        missing_data = data.isnull().sum()
        missing_percent = (missing_data / len(data)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
        else:
            print("No missing values found!")
        
        # Duplicate rows
        duplicates = data.duplicated().sum()
        print(f"\n2. Duplicate Rows: {duplicates}")
        if duplicates > 0:
            print(f"   ({duplicates/len(data)*100:.2f}% of total data)")
        
        # Data types summary
        print(f"\n3. Data Types Summary:")
        dtype_counts = data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
        
        # Memory usage
        print(f"\n4. Memory Usage:")
        memory_usage = data.memory_usage(deep=True).sum() / 1024**2  # Convert to MB
        print(f"   Total: {memory_usage:.2f} MB")
    
    def get_column_summary(self, data=None):
        """
        Get summary statistics for all columns
        Args:
            data (pd.DataFrame): Dataset to analyze (uses train_data if None)
        """
        if data is None:
            data = self.train_data
            
        if data is None:
            print("No data available. Please load data first.")
            return
        
        # Numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print("\n=== NUMERICAL COLUMNS SUMMARY ===")
            print(data[numerical_cols].describe())
        
        # Categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print("\n=== CATEGORICAL COLUMNS SUMMARY ===")
            for col in categorical_cols:
                print(f"\n{col}:")
                print(f"  Unique values: {data[col].nunique()}")
                if data[col].nunique() <= 10:  # Show values if not too many
                    print(f"  Values: {list(data[col].unique())}")
                print(f"  Top 5 values:")
                print(data[col].value_counts().head().to_string())


def main():
    """Main function to demonstrate DataLoader usage"""
    print("=== CAR INSURANCE CLAIM DATA LOADER ===\n")
    
    # Create DataLoader instance
    loader = DataLoader()
    
    # List available files
    loader.list_data_files()
    
    # Load training data
    print("\n" + "="*50)
    train_data = loader.load_train_data()
    
    if train_data is not None:
        # Get basic information
        loader.get_basic_info()
        
        # Check data quality
        loader.check_data_quality()
        
        # Get column summary
        loader.get_column_summary()
    
    print("\n=== DATA LOADING COMPLETED ===")


if __name__ == "__main__":
    main()
