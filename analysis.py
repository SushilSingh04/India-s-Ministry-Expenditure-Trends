# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for visualizations
sns.set_style('whitegrid')
df = pd.read_csv('Pay_govt_ministry.csv')

# Initial inspection
# print("Data shape:", df.shape)

# print("\nFirst 5 rows:")
# print(df.head())

# print("\nData summary:")
# print(df.info())

# print("\nDescriptive statistics:")
# print(df.describe(include='all'))

# print("\nMissing values:")
# print(df.isnull().sum())

# Extract year from the 'Year' column
df['Year'] = df['Year'].str.extract(r'(\d{4})').astype(int)

#Change all column names having (UOM:INR(IndianRupees)) to simpler names
df.rename(columns={
    col: col.split(' (UOM:INR(IndianRupees))')[0].strip()
    for col in df.columns
    if '(UOM:INR(IndianRupees))' in col
}, inplace=True)

# Drop unnecessary columns (e.g., 'Additional Info' with many NaN values)
df.drop(columns=['Additional Info', 'Country'], inplace=True)

# Standardize ministry/department names (example)
df['Ministry Or Department'] = df['Ministry Or Department'].str.strip().str.upper()

# Check for duplicates
df.drop_duplicates(inplace=True)

# # Handle missing values (drop rows with critical NaN values)
df.dropna(subset=['Pay'], inplace=True)

# # Summary statistics
print(df.describe())

# # Yearly trends in Pay, DA, and HRA
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=df, x='Year', y='Pay', label='Pay')
# sns.lineplot(data=df, x='Year', y='Dearness Allowance', label='DA')
# sns.lineplot(data=df, x='Year', y='House Rent Allowance', label='HRA')
# plt.title('Yearly Trends in Pay, DA, and HRA')
# plt.show()

# # Top 10 ministries by total pay
# top_ministries = df.groupby('Ministry Or Department')['Pay'].sum().nlargest(10)
# plt.figure(figsize=(10, 6))
# sns.barplot(x=top_ministries.values, y=top_ministries.index)
# plt.title('Top 10 Ministries by Total Pay')
# plt.xlabel('Total Pay (INR)')
# plt.show()

# # Correlation matrix
# corr_matrix = df.select_dtypes(include=np.number).corr() #selects all numneric columns
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()

# Select numerical columns for pairplot (adjust based on your focus)
pairplot_cols = ['Pay', 'Dearness Allowance', 'House Rent Allowance', 'Transport Allowance', 'Bonus']

# Create a pairplot
plt.figure(figsize=(15, 10))
sns.pairplot(df[pairplot_cols], corner=True, diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pairwise Relationships Between Key Financial Components', y=1.02)
plt.show()

# # Descriptive statistics for key columns
# print(df[['Pay ', 'Dearness Allowance']].describe())

# # Hypothesis test: Compare Pay between two years (e.g., 2020 vs. 2010)
# pay_2020 = df[df['Year'] == 2020]['Pay']
# pay_2010 = df[df['Year'] == 2010]['Pay']
# t_stat, p_value = stats.ttest_ind(pay_2020, pay_2010, nan_policy='omit')
# print(f"T-statistic: {t_stat}, P-value: {p_value}")

# # Distribution of Pay across years
# plt.figure(figsize=(14, 7))
# sns.boxplot(data=df, x='Year', y='Pay')
# plt.xticks(rotation=45)
# plt.title('Distribution of Pay Across Years')
# plt.show()

# df.to_csv('cleaned_ministry_data.csv', index=False)