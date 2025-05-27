import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


df = pd.read_csv(r'C:\Users\Anshu2006\OneDrive\Desktop\SOURCE_FILE_Anshuman_Patel_500122770+MehakDahiya_500119604\House1_Price_Multiplied.csv')



# Set a nice style
sns.set(style="whitegrid")

# 1. Distribution of House Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 2. Interactive Price vs Area Scatter Plot with OLS Trendline
fig = px.scatter(
    df,
    x='Area',
    y='Price',
    hover_data=['Bedrooms', 'Bathrooms', 'Location'],
    title="Interactive House Price vs Area (with OLS Trendline)",
    color='Location',
    trendline="ols"
)
fig.show()



# 3. Price Variation vs Number of Bathrooms
plt.figure(figsize=(10, 6))
sns.boxplot(x='Bathrooms', y='Price', data=df, palette='coolwarm')
plt.title('Price Variation vs Number of Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

# 4. House Price by State
plt.figure(figsize=(12, 6))
avg_price_state = df.groupby('State')['Price'].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_state.index, y=avg_price_state.values, palette='Spectral')
plt.xticks(rotation=45)
plt.title(' House Price vs State')
plt.ylabel(' Price')
plt.xlabel('State')
plt.tight_layout()
plt.show()

# 5.  House Price vs Number of Bathrooms
plt.figure(figsize=(10, 6))
avg_price_bathroom = df.groupby('Bathrooms')['Price'].mean()
sns.barplot(x=avg_price_bathroom.index, y=avg_price_bathroom.values, palette='Set1')
plt.title(' House Price vs Number of Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel(' Price')
plt.tight_layout()
plt.show()

# 6. Location vs Price Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Location', y='Price', data=df, palette='Pastel1')
plt.title('Location vs House Price')
plt.xlabel('Location')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. RoomsPerFloor vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='RoomsPerFloor', y='Price', data=df, hue='GarageBinary', palette='cool')
plt.title('Rooms per Floor vs Price (colored by Garage)')
plt.xlabel('Rooms per Floor')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

# 8. AreaPerRoom vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AreaPerRoom', y='Price', data=df, hue='GarageBinary', palette='cubehelix')
plt.title('Area per Room vs Price (colored by Garage)')
plt.xlabel('Area per Room (sq ft)')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

# 9. Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
