import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


df = pd.read_csv(r'C:\Users\Anshu2006\OneDrive\Desktop\SOURCE_FILE_Anshuman_Patel_500122770+MehakDahiya_500119604\House1_Price_Multiplied.csv')

# ---------------- Feature Extraction ----------------

# 1. House Age
current_year = datetime.now().year
df['Age'] = current_year - df['YearBuilt']

# 2. Rooms per Floor
df['RoomsPerFloor'] = (df['Bedrooms'] + df['Bathrooms']) / df['Floors'].replace(0, 1)

# 3. Area per Room
df['AreaPerRoom'] = df['Area'] / (df['Bedrooms'] + df['Bathrooms']).replace(0, 1)


# Save the updated CSV
output_file = r'C:\Users\Anshu2006\OneDrive\Desktop\DATA DCIENCE PROJECT\House1_Price_WithFeatures.csv'
df.to_csv(output_file, index=False)

print(f"Updated CSV with features saved to {output_file}")


# Set a nice style
sns.set(style="whitegrid")

# Plot 1: Distribution of Area
plt.figure(figsize=(8,5))
sns.histplot(df['Area'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Area')
plt.xlabel('Area (sq ft)')
plt.ylabel('Count')
plt.show()

# Plot 2: Distribution of Age
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True, color='lightgreen')
plt.title('Distribution of House Age')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.show()

# Plot 3: Rooms Per Floor
plt.figure(figsize=(8,5))
sns.histplot(df['RoomsPerFloor'], bins=30, kde=True, color='salmon')
plt.title('Distribution of Rooms per Floor')
plt.xlabel('Rooms per Floor')
plt.ylabel('Count')
plt.show()


