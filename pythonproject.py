# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 01:29:26 2025

@author: hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


# Set aesthetics for plots
sns.set_theme(style="darkgrid", palette="flare")
#plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({
    'figure.facecolor': '#1c1c1c',
    'axes.facecolor': '#2a2a2a',
    'savefig.facecolor': '#1c1c1c',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'axes.titlecolor': 'white'
})


# Load the dataset
file_path = r"C:/Users/hp/Desktop/realtime_data.csv"
df = pd.read_csv(file_path)


# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Convert date
df['lastUpdated'] = pd.to_datetime(df['lastUpdated'], format='%d-%m-%Y', errors='coerce')

# Drop duplicates
df = df.drop_duplicates()

# Total beneficiaries by state
state_beneficiaries = df.groupby('state_name')['total_beneficiaries'].sum().reset_index().sort_values(by='total_beneficiaries', ascending=False)

# Bar plot (updated hue to fix warning)
plt.figure(figsize=(14, 6))
sns.barplot(data=state_beneficiaries, x='state_name', y='total_beneficiaries', hue='state_name', palette='Spectral', dodge=False)
plt.xticks(rotation=90)
plt.title("Total Beneficiaries by State")
plt.xlabel("State")
plt.ylabel("Total Beneficiaries")
plt.legend([], [], frameon=False)
plt.tight_layout()
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(10, 7))
correlation = df[['total_beneficiaries', 'total_aadhar', 'total_mobileno', 'total_sc', 'total_st', 'total_gen', 'total_obc']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()




#after this no graph is working 
# Scatterplot - Aadhaar vs Beneficiaries (clean)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='total_aadhar', y='total_beneficiaries', hue='state_name', palette='tab20', alpha=0.7)
plt.title("Aadhaar Linked vs Total Beneficiaries")
plt.xlabel("Aadhaar Linked")
plt.ylabel("Total Beneficiaries")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
plt.tight_layout()
plt.show()

# Correlation between aadhar and beneficiaries
cor_val = df['total_aadhar'].corr(df['total_beneficiaries'])
print(f"\nCorrelation between Total Aadhar and Total Beneficiaries: {cor_val}")


# Objective 1: Inclusion Score = avg of mobile + aadhar coverage
df['inclusion_score'] = (df['total_aadhar'] + df['total_mobileno']) / (2 * df['total_beneficiaries'])
df['inclusion_score'] = df['inclusion_score'].clip(upper=1.0)

# Objective 2: Top 5 inclusive districts
top_districts = df.groupby('district_name')['inclusion_score'].mean().reset_index().sort_values(by='inclusion_score', ascending=False).head(5)
print("\nTop 5 Districts by Inclusion Score:\n", top_districts)

# Objective 3: Top 5 inclusive states
top_states = df.groupby('state_name')['inclusion_score'].mean().reset_index().sort_values(by='inclusion_score', ascending=False).head(5)
print("\nTop 5 States by Inclusion Score:\n", top_states)

# Plot top 5 states with fixed hue
plt.figure(figsize=(10, 5))
sns.barplot(data=top_states, x='state_name', y='inclusion_score', hue='state_name', palette='coolwarm', dodge=False)
plt.title("Top 5 States by Inclusion Score")
plt.ylabel("Inclusion Score")
plt.xlabel("State")
plt.legend([], [], frameon=False)
plt.tight_layout()
plt.show()

# Objective 4: Trend in scheme reporting over time
monthly_report = df.copy()
monthly_report['month'] = monthly_report['lastUpdated'].dt.to_period('M')
monthly_trend = monthly_report.groupby('month').size().reset_index(name='report_count')


# Export static plots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['total_beneficiaries', 'total_aadhar', 'total_mobileno']], palette='Set3')
plt.title("Boxplot for Key Metrics")
plt.tight_layout()
plt.savefig("/Users/akshat/Desktop/key_metrics_boxplot.pdf")
plt.show()
