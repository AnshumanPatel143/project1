import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from the given path
file_path = r'C:\Users\Anshu2006\OneDrive\Desktop\SOURCE_FILE_Anshuman_Patel_500122770+MehakDahiya_500119604\House1_Price_Multiplied.csv'
df = pd.read_csv(file_path)

# Analyzing the data
num_records = df.shape[0]  # Number of records (rows)
num_features = df.shape[1] - 1  # Number of features (columns), excluding the target variable
target_variable = 'HousePrice'  # Assuming 'HousePrice' is the target variable

# Identifying types of features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
temporal_features = df.select_dtypes(include=['datetime']).columns.tolist()

# Additional analysis
# 1. Check for missing values
missing_values = df.isnull().sum()

# 2. Summary statistics for numerical columns
summary_statistics = df[numerical_features].describe()

# 3. Correlation matrix (for numerical features)
correlation_matrix = df[numerical_features].corr()

# 4. Value counts for categorical features
categorical_value_counts = {feature: df[feature].value_counts() for feature in categorical_features}

# Displaying results
print(f"Total number of records: {num_records}")
print(f"Total number of features (excluding target): {num_features}")
print(f"Target variable: {target_variable}")
print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")
print(f"Temporal features: {temporal_features}")

# 1. Missing values
print("\nMissing values per feature:")
print(missing_values)

# 2. Summary statistics
print("\nSummary statistics (for numerical features):")
print(summary_statistics)


# 4. Value counts for categorical features
print("\nValue counts for categorical features:")
for feature, counts in categorical_value_counts.items():
    print(f"\n{feature} value counts:")
    print(counts)

# Data types of all features
print("\nData types of features:")
print(df.dtypes)
