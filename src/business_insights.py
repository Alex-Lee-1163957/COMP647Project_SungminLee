"""
Business Insights and Research Questions
Course: COMP647 - Machine Learning
Student: Sungmin Lee (1163957)
Purpose: Explore business-relevant questions based on real-world insurance experience
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BusinessInsights:
    """
    Generate business insights from car insurance data based on real-world experience
    """
    
    def __init__(self, data, target_column):
        """
        Initialize with processed dataset
        Args:
            data (pd.DataFrame): Cleaned dataset
            target_column (str): Target variable name
        """
        self.data = data
        self.target_column = target_column
        
    def analyze_high_risk_profiles(self):
        """
        Identify customer profiles most likely to file claims
        
        Business Context: In my daily work at Allied Financial, I notice certain 
        customer patterns when they call to file claims. This analysis helps 
        identify these patterns systematically.
        """
        print("\n" + "="*60)
        print("BUSINESS QUESTION 1: Which customer profiles are highest risk?")
        print("="*60)
        
        # Get claim rates by different customer segments
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        print("\nFrom my experience handling claims, I've observed patterns in:")
        print("- Age groups and driving experience")
        print("- Vehicle types and their claim frequencies") 
        print("- Geographic and demographic factors")
        print("\nLet's validate these observations with data:")
        
        # Analyze age groups
        if 'age' in self.data.columns:
            # Create age groups based on typical insurance categories
            age_bins = [0, 25, 35, 50, 65, 100]
            age_labels = ['Under 25', '25-34', '35-49', '50-64', '65+']
            self.data['age_group'] = pd.cut(self.data['age'], bins=age_bins, labels=age_labels)
            
            age_claims = self.data.groupby('age_group')[self.target_column].agg(['count', 'sum', 'mean'])
            age_claims.columns = ['Total_Customers', 'Claims_Filed', 'Claim_Rate']
            age_claims['Claim_Rate'] = age_claims['Claim_Rate'] * 100
            
            print(f"\nClaim rates by age group:")
            print(age_claims.round(2))
            
            # Business insight
            highest_risk_age = age_claims['Claim_Rate'].idxmax()
            print(f"\nBusiness Insight: {highest_risk_age} age group has the highest claim rate")
            print("This aligns with my experience - these customers often need more guidance")
            print("when filing claims and require more detailed documentation.")
        
        # Analyze vehicle types if available
        vehicle_cols = [col for col in self.data.columns if 'vehicle' in col.lower() or 'car' in col.lower()]
        if vehicle_cols:
            vehicle_col = vehicle_cols[0]
            vehicle_claims = self.data.groupby(vehicle_col)[self.target_column].agg(['count', 'sum', 'mean'])
            vehicle_claims.columns = ['Total_Customers', 'Claims_Filed', 'Claim_Rate']
            vehicle_claims['Claim_Rate'] = vehicle_claims['Claim_Rate'] * 100
            vehicle_claims = vehicle_claims.sort_values('Claim_Rate', ascending=False)
            
            print(f"\nClaim rates by vehicle type:")
            print(vehicle_claims.head().round(2))
            
            highest_risk_vehicle = vehicle_claims['Claim_Rate'].idxmax()
            print(f"\nBusiness Insight: {highest_risk_vehicle} owners have highest claim rates")
            print("In my work, I notice these vehicles often have more complex claims")
            print("requiring additional documentation and longer processing times.")
    
    def analyze_claim_severity_predictors(self):
        """
        Identify factors that predict claim complexity
        
        Business Context: Some claims are straightforward (fender benders), 
        others require extensive documentation (total loss, injury). Understanding
        predictors helps me prepare appropriate paperwork in advance.
        """
        print("\n" + "="*60) 
        print("BUSINESS QUESTION 2: What predicts claim complexity?")
        print("="*60)
        
        print("\nIn my daily work, claim complexity varies significantly:")
        print("- Simple claims: Minor damage, clear fault, standard paperwork")
        print("- Complex claims: Multiple parties, injuries, disputed fault")
        print("\nAnalyzing factors that might predict complexity:")
        
        # Look for factors that correlate with claims
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != self.target_column]
        
        correlations = []
        for col in numerical_cols:
            corr = self.data[col].corr(self.data[self.target_column])
            correlations.append({'Feature': col, 'Correlation': corr})
        
        corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
        print(f"\nFeatures most correlated with claims:")
        print(corr_df.head(10).round(3))
        
        # Business insights
        top_predictor = corr_df.iloc[0]
        print(f"\nBusiness Insight: {top_predictor['Feature']} shows strongest correlation")
        print("This information helps me anticipate the type of documentation needed")
        print("when a customer with these characteristics calls to file a claim.")
    
    def analyze_proactive_service_opportunities(self):
        """
        Identify opportunities for proactive customer service
        
        Business Context: Instead of waiting for customers to call after accidents,
        we could proactively reach out to high-risk customers with safety tips,
        policy reviews, or preventive measures.
        """
        print("\n" + "="*60)
        print("BUSINESS QUESTION 3: How can we improve proactive service?")
        print("="*60)
        
        print("\nCurrent reactive approach:")
        print("1. Customer has accident → 2. Customer calls → 3. I process claim")
        print("\nProposed proactive approach:")
        print("1. Identify high-risk customers → 2. Proactive outreach → 3. Prevention/preparation")
        
        # Identify high-risk customer segments
        claim_rate = self.data[self.target_column].mean() * 100
        print(f"\nOverall claim rate: {claim_rate:.1f}%")
        
        # Create risk segments
        high_risk_threshold = claim_rate * 1.5  # 50% above average
        
        # Find combinations of features that create high-risk segments
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) >= 2:
            col1, col2 = categorical_cols[:2]
            risk_segments = self.data.groupby([col1, col2])[self.target_column].agg(['count', 'mean'])
            risk_segments.columns = ['Customer_Count', 'Claim_Rate']
            risk_segments['Claim_Rate'] = risk_segments['Claim_Rate'] * 100
            
            # Filter for segments with meaningful sample size
            risk_segments = risk_segments[risk_segments['Customer_Count'] >= 10]
            high_risk_segments = risk_segments[risk_segments['Claim_Rate'] > high_risk_threshold]
            
            if len(high_risk_segments) > 0:
                print(f"\nHigh-risk customer segments (>{high_risk_threshold:.1f}% claim rate):")
                print(high_risk_segments.sort_values('Claim_Rate', ascending=False).round(1))
                
                print(f"\nBusiness Opportunity:")
                print("These segments could benefit from:")
                print("- Quarterly safety check-ins")
                print("- Pre-prepared claim documentation templates")
                print("- Direct relationships with preferred repair shops")
                print("- Enhanced coverage recommendations")
    
    def generate_operational_recommendations(self):
        """
        Generate specific operational improvements based on data insights
        
        Business Context: Translate data findings into actionable business processes
        that improve customer service and operational efficiency.
        """
        print("\n" + "="*60)
        print("BUSINESS QUESTION 4: What operational improvements can we make?")
        print("="*60)
        
        print("\nBased on data analysis and my operational experience:")
        
        # Calculate some key metrics
        total_customers = len(self.data)
        total_claims = self.data[self.target_column].sum()
        claim_rate = (total_claims / total_customers) * 100
        
        print(f"\nKey Metrics:")
        print(f"- Total customers analyzed: {total_customers:,}")
        print(f"- Total claims: {total_claims:,}")
        print(f"- Overall claim rate: {claim_rate:.2f}%")
        
        print(f"\nOperational Recommendations:")
        
        print(f"\n1. RISK-BASED CUSTOMER SEGMENTATION")
        print(f"   - Implement automated risk scoring for new customers")
        print(f"   - Create different service protocols for high/low risk segments")
        print(f"   - Allocate more experienced staff to high-risk customer calls")
        
        print(f"\n2. PROACTIVE CLAIM PREPARATION")
        print(f"   - Pre-populate claim forms for high-risk customers")
        print(f"   - Maintain preferred vendor relationships by customer segment")
        print(f"   - Create customer-specific documentation checklists")
        
        print(f"\n3. PROCESS EFFICIENCY IMPROVEMENTS") 
        print(f"   - Develop claim complexity prediction models")
        print(f"   - Implement automated initial claim assessment")
        print(f"   - Create fast-track processes for predictable claim types")
        
        print(f"\n4. CUSTOMER COMMUNICATION OPTIMIZATION")
        print(f"   - Send targeted safety tips to high-risk segments")
        print(f"   - Provide claim process education before incidents occur")
        print(f"   - Offer premium discounts for safety course completion")
        
        print(f"\nExpected Benefits:")
        print(f"- Reduced claim processing time")
        print(f"- Improved customer satisfaction")
        print(f"- Lower operational costs")
        print(f"- Better risk management")


def main():
    """
    Run business insights analysis
    """
    print("CAR INSURANCE BUSINESS INSIGHTS ANALYSIS")
    print("Based on Real-World Insurance Claim Processing Experience")
    print("Student: Sungmin Lee (1163957)")
    print("Position: Financial Supporter, Allied Financial, Christchurch NZ")
    
    # This would typically load the processed data from main analysis
    # For demonstration, we'll create a simple example
    print("\nNote: This analysis should be run after data preprocessing")
    print("Run this module from main.py after EDA completion")
    
    return None

if __name__ == "__main__":
    main()
