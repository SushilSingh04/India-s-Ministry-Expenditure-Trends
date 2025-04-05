# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for visualizations
sns.set_style('whitegrid')
# Load dataset (example using Titanic dataset)
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
df = pd.read_csv(url)

# Initial inspection
print("Data shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData summary:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe(include='all'))
print("\nMissing values:")
print(df.isnull().sum())
# Handle missing values
# Fill age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Drop cabin column (too many missing values)
df.drop('Cabin', axis=1, inplace=True)

# Drop remaining missing values (if any)
df.dropna(inplace=True)

# Check for duplicates
print("Duplicate rows:", df.duplicated().sum())

# Handle outliers (example for Fare)
z_scores = stats.zscore(df['Fare'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
df = df[filtered_entries]

# Convert categorical variables
df['Sex'] = df['Sex'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')
# Univariate analysis
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()

# Bivariate analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Age vs Fare colored by Survival')
plt.show()

# Multivariate analysis
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# Survival rate by gender
survival_gender = pd.crosstab(df['Sex'], df['Survived'])
print(survival_gender)
chi2, p, dof, expected = stats.chi2_contingency(survival_gender)
print(f"\nChi-square test p-value: {p:.4f}")

# Age difference between survivors and non-survivors
survived_age = df[df['Survived'] == 1]['Age']
not_survived_age = df[df['Survived'] == 0]['Age']
t_stat, p_val = stats.ttest_ind(survived_age, not_survived_age)
print(f"\nT-test p-value: {p_val:.4f}")

# Fare vs Pclass relationship
fare_class = df.groupby('Pclass')['Fare'].agg(['mean', 'median', 'std'])
print("\nFare statistics by Class:")
print(fare_class)
# Survival rate by class and gender
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df)
plt.title('Survival Rate by Class and Gender')
plt.ylabel('Survival Rate')
plt.show()

# Distribution of fares by survival
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare Distribution by Survival Status')
plt.show()

# Pairplot for numerical variables
sns.pairplot(df[['Age', 'Fare', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Survived']], 
             hue='Survived')
plt.suptitle('Pairplot of Numerical Variables', y=1.02)
plt.show()
# Save cleaned data
df.to_csv('cleaned_titanic.csv', index=False)

# Document key findings
print("\nKey Insights:")
print("- Younger passengers had higher survival rates")
print("- Women and children were prioritized in rescue efforts")
print("- Higher fare classes (1st class) had better survival rates")
print("- Strong correlation between fare and passenger class")