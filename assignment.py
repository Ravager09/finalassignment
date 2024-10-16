import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data
df = pd.read_csv('content/customer_churn_dataset-training-master.csv')

# Display basic information about the dataset
print(df.info())

# Display summary statistics
print(df.describe())

plt.figure(figsize=(20, 20))

# 1. Pie chart: Distribution of Subscription Types
plt.subplot(2, 3, 1)
df['Subscription Type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Subscription Types')
plt.tight_layout()
plt.show()

# 2. Scatter plot: Usage Frequency vs Total Spend
plt.subplot(2, 3, 2)
plt.scatter(df['Usage Frequency'], df['Total Spend'])
plt.xlabel('Usage Frequency')
plt.ylabel('Total Spend')
plt.title('Usage Frequency vs Total Spend')
plt.tight_layout()
plt.show()

# 3. Box plot: Total Spend by Subscription Type
plt.subplot(2, 3, 3)
sns.boxplot(x='Subscription Type', y='Total Spend', data=df)
plt.title('Total Spend by Subscription Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Histogram: Age Distribution
plt.subplot(2, 3, 4)
df['Age'].hist(bins=20)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.tight_layout()
plt.show()

# 5. Bar plot: Churn Rate by Contract Length
plt.subplot(2, 3, 5)
churn_rate = df.groupby('Contract Length')['Churn'].mean()
churn_rate.plot(kind='bar')
plt.xlabel('Contract Length')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Contract Length')
plt.tight_layout()
plt.show()

# 6. Stacked bar chart: Churn by Subscription Type and Contract Length
plt.subplot(2, 2, 3)
churn_by_sub_contract = df.groupby(['Subscription Type', 'Contract Length'])['Churn'].mean().unstack()
churn_by_sub_contract.plot(kind='bar', stacked=True)
plt.title('Churn Rate by Subscription Type and Contract Length')
plt.xlabel('Subscription Type')
plt.ylabel('Churn Rate')
plt.legend(title='Contract Length')
plt.tight_layout()
plt.show()

#7. Violin plot: Distribution of Total Spend by Churn
plt.subplot(2, 2, 4)
sns.violinplot(x='Churn', y='Total Spend', data=df)
plt.title('Distribution of Total Spend by Churn')
plt.show()


# 5. Grouped bar chart: Average Usage Frequency and Support Calls by Churn
plt.figure(figsize=(12, 6))
df_grouped = df.groupby('Churn')[['Usage Frequency', 'Support Calls']].mean()
df_grouped.plot(kind='bar', ylabel='Average Value')
plt.title('Average Usage Frequency and Support Calls by Churn')
plt.legend(['Usage Frequency', 'Support Calls'])
plt.xticks([0, 1], ['Not Churned', 'Churned'], rotation=0)
plt.tight_layout()
plt.show()

# 6. Scatter plot: Tenure vs Total Spend, colored by Churn
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Tenure', y='Total Spend', hue='Churn', style='Churn')
plt.title('Tenure vs Total Spend, colored by Churn')
plt.tight_layout()
plt.show()


# Select relevant numerical variables
variables = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Churn']

# Create the scatter plot matrix
plt.figure(figsize=(20, 20))
sns.pairplot(df[variables], hue='Churn', diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Scatter Plot Matrix of Key Variables', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# Additional statistical analysis
print("\nChurn Rate:")
print(df['Churn'].mean())

print("\nAverage Tenure:")
print(df['Tenure'].mean())

print("\nT-test: Total Spend for Churned vs Non-Churned Customers")
churned = df[df['Churn'] == 1]['Total Spend']
non_churned = df[df['Churn'] == 0]['Total Spend']
t_stat, p_value = stats.ttest_ind(churned, non_churned)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")