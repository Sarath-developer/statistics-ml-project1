
# ============================================
# PRACTICE: STATISTICS IN ML (FULL WORKING CODE)
# ============================================

# STEP 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# STEP 2: LOAD DATA
df = pd.read_csv("students_data.csv")

print("First 5 rows:")
print(df.head())


# STEP 3: BASIC INFO
print("\nShape:", df.shape)
print("\nColumns:", df.columns)

print("\nInfo:")
df.info()


# STEP 4: CONVERT CATEGORICAL → NUMERICAL
# gender: Male=0, Female=1
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})


# STEP 5: DESCRIPTIVE STATISTICS
print("\nStatistical Summary:")
print(df.describe())

print("\nMean:")
print(df.mean())

print("\nMedian:")
print(df.median())

print("\nMode:")
print(df.mode().iloc[0])


# STEP 6: MEASURE OF DISPERSION
print("\nVariance:")
print(df.var())

print("\nStandard Deviation:")
print(df.std())

print("\nRange:")
print(df.max() - df.min())


# STEP 7: PERCENTILES & QUARTILES
print("\nPercentiles:")
print(df.quantile([0.25, 0.5, 0.75]))

print("\n5 Number Summary:")
print(df.describe().loc[['min','25%','50%','75%','max']])


# STEP 8: MISSING VALUES
print("\nMissing Values:")
print(df.isnull().sum())


# STEP 9: HISTOGRAM
df.hist(figsize=(10,8))
plt.suptitle("Histogram - Distribution")
plt.show()


# STEP 10: BOXPLOT
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Boxplot - Outliers")
plt.show()


# STEP 11: CORRELATION
correlation = df.corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(10,6))
sns.heatmap(correlation, annot=True)
plt.title("Correlation Heatmap")
plt.show()


# STEP 12: COVARIANCE
covariance = df.cov()

print("\nCovariance Matrix:")
print(covariance)


# STEP 13: SKEWNESS & KURTOSIS
print("\nSkewness:")
print(df.skew())

print("\nKurtosis:")
print(df.kurt())


# STEP 14: OUTLIER DETECTION (IQR METHOD)

# Use only numeric columns
numeric_df = df.select_dtypes(include=np.number)

Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)

IQR = Q3 - Q1

outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | 
            (numeric_df > (Q3 + 1.5 * IQR)))

print("\nOutliers (True = Outlier):")
print(outliers)


# STEP 15: DISTRIBUTION PLOT
plt.figure(figsize=(8,5))
sns.histplot(df['final_score'], kde=True)
plt.title("Final Score Distribution")
plt.show()
