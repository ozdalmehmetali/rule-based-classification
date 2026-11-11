# Setup and Data Preparation
# In the first step, we import the necessary libraries, configure pandas display settings, and load our dataset.
# We also drop columns like Customer ID and Promo Code Used as they do not provide value for this specific analysis.
import pandas as pd
import numpy as np

# Pandas display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

# Load the dataset
df = pd.read_csv('/kaggle/input/datasets/iamsouravbanerjee/customer-shopping-trends-dataset/shopping_trends.csv')
# Drop columns not needed for the analysis
df.drop(["Customer ID", "Promo Code Used"], axis=1, inplace=True)

# Initial look at the dataset
print(df.shape)
print(df.info())
df.head()

# 2. Exploratory Data Analysis (EDA)
# To understand the general structure of the dataset, we examine category frequencies,
# revenue distributions based on locations, and location-category breakdowns.

# Number of unique categories and their frequencies
print("Number of Unique Categories:", df["Category"].nunique())
print(df["Category"].value_counts())

# Total revenue distribution by location
print(df.groupby("Location").agg({"Purchase Amount (USD)": "sum"}).sort_values(by="Purchase Amount (USD)", ascending=False))

# Average purchase amounts grouped by Location and Category
print(df.groupby(["Location", "Category"]).agg({"Purchase Amount (USD)": "mean"}))


#3. Feature Engineering
# To perform rule-based classification,
# we need to convert the continuous Age variable into a categorical variable by dividing it into logical bins.

# Calculate average purchase amounts broken down by Location, Category, Gender, and Age
agg_df = df.groupby(["Location", "Category", "Gender", "Age"]).agg({"Purchase Amount (USD)": "mean"}).sort_values(by="Purchase Amount (USD)", ascending=False)
agg_df = agg_df.reset_index()

# Categorize the Age variable (Age_Cat)
# Note: The starting bin is set to 17 to include 18-year-olds.
bins = [17, 25, 35, 45, 70]
labels = ["18_25", "26_35", "36_45", "46_70"]
agg_df["Age_Cat"] = pd.cut(agg_df["Age"], bins=bins, labels=labels)

agg_df.head()

# 4. Creating Level-Based Personas
# We combine demographic features and shopping habits to create unique customer personas
# (e.g., NEW YORK_CLOTHING_MALE_26_35).
# We then group the dataset by these new personas to find the average purchase amount for each specific profile.

# Create persona names using list comprehension
# When using agg_df.values, the indices are:
# 0: Location, 1: Category, 2: Gender, 5: Age_Cat
agg_df["CUSTOMER_LEVEL_BASED"] = [f"{row[0]}_{row[1]}_{row[2]}_{row[5]}".upper() for row in agg_df.values]

# Aggregate personas and calculate their average purchase amounts
agg_df = agg_df.groupby("CUSTOMER_LEVEL_BASED").agg({"Purchase Amount (USD)": "mean"}).reset_index()

agg_df.head()

# 5. Segmentation of Personas
# We divide these unique personas into 4 core segments
# (Copper, Bronze, Silver, Gold) based on their average revenue contribution (Purchase Amount) using quartiles.

# Create 4 different segments based on the purchase amount
agg_df["SEGMENT"] = pd.qcut(agg_df["Purchase Amount (USD)"], 4, labels=["copper", "bronze", "silver", "gold"])

# Statistical summary of the segments
agg_df.groupby("SEGMENT", observed=True).agg({"Purchase Amount (USD)": ["mean", "max", "sum"]})

# 6. Prediction for New Customers
# Our rule-based model is now ready.
# By looking at the demographic features of a new incoming customer,
# we can predict which segment they belong to and the average revenue they are expected to generate for the company.

# Example 1: A 33-year-old female customer from New York purchasing from the Clothing category
new_user_1 = "NEW YORK_CLOTHING_FEMALE_26_35"
print("Customer Profile 1:")
print(agg_df[agg_df["CUSTOMER_LEVEL_BASED"] == new_user_1])

print("-" * 50)

# Example 2: A 25-year-old male customer from California purchasing from the Accessories category
new_user_2 = "CALIFORNIA_ACCESSORIES_MALE_18_25"
print("Customer Profile 2:")
print(agg_df[agg_df["CUSTOMER_LEVEL_BASED"] == new_user_2])

