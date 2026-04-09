# ===============================
#  IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
#  LOAD DATA
# ===============================
df = pd.read_csv("data.csv")

print("\n📊 DATA PREVIEW:")
print(df.head())

# ===============================
#  BASIC STATISTICS (Describe)
# ===============================
print("\n📈 STATISTICAL SUMMARY:")
print(df.describe())

# ===============================
#  CENTRAL TENDENCY
# ===============================
print("\n CENTRAL TENDENCY (Math Scores)")
print("Mean:", df["Math"].mean())
print("Median:", df["Math"].median())
print("Mode:", df["Math"].mode()[0])

# ===============================
#  DISPERSION
# ===============================
print("\n DISPERSION")
print("Variance:", df["Math"].var())
print("Standard Deviation:", df["Math"].std())

# ===============================
#  WHY n-1 (Sample Variance)
# ===============================
n = len(df["Math"])
sample_variance = df["Math"].var()
population_variance = df["Math"].var(ddof=0)

print("\n VARIANCE COMPARISON")
print("Sample Variance (n-1):", sample_variance)
print("Population Variance (n):", population_variance)

# ===============================
#  VARIABLES & RANDOM VARIABLES
# ===============================
print("\n VARIABLES")
print("Independent Variable: Hours_Study")
print("Dependent Variable: Math")

# ===============================
#  HISTOGRAM
# ===============================
plt.hist(df["Math"], bins=5)
plt.title("Histogram of Math Scores")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()

# ===============================
#  PERCENTILES & QUARTILES
# ===============================
print("\n QUARTILES")
print("25%:", np.percentile(df["Math"], 25))
print("50% (Median):", np.percentile(df["Math"], 50))
print("75%:", np.percentile(df["Math"], 75))

# ===============================
#  5 NUMBER SUMMARY
# ===============================
print("\n 5 NUMBER SUMMARY")
print("Min:", df["Math"].min())
print("Q1:", df["Math"].quantile(0.25))
print("Median:", df["Math"].median())
print("Q3:", df["Math"].quantile(0.75))
print("Max:", df["Math"].max())

# ===============================
#  CORRELATION
# ===============================
print("\n CORRELATION MATRIX")
print(df.corr())

# ===============================
#  COVARIANCE
# ===============================
print("\n COVARIANCE")
print(df.cov())

# ===============================
# SCATTER PLOT (ML BASE)
# ===============================
plt.scatter(df["Hours_Study"], df["Math"])
plt.xlabel("Hours Study")
plt.ylabel("Math Score")
plt.title("Study vs Score")
plt.show()
