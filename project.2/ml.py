import pandas as pd

# Load dataset
df = pd.read_csv(
    r"C:\Users\l\OneDrive\Desktop\programming\file python\data_analysis.csv"
)

# DATA CLEANING
# Remove $ and commas, then convert to float

df["Unit Price"] = (
    df["Unit Price"]
    .replace("[$,]", "", regex=True)
    .astype(float)
)

df["Sales"] = (
    df["Sales"]
    .replace("[$,]", "", regex=True)
    .astype(float)
)

df["Cost"] = (
    df["Cost"]
    .replace("[$,]", "", regex=True)
    .astype(float)
)

# Check data types
print(df.dtypes)

# Select input features
X = df[[
    "Quantity",
    "Unit Price",
    "Cost",
    "ProductKey",
    "SalesTerritoryKey"
]]

# Target variable
y = df["Sales"]

print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))

# MODEL TRAINING
from sklearn.linear_model import LinearRegression

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("Model Trained Successfully")

# PREDICTION & EVALUATION
from sklearn.metrics import mean_absolute_error, r2_score

# Predict sales
predictions = model.predict(X_test)

# Accuracy score
r2 = r2_score(y_test, predictions)

# Error calculation
mae = mean_absolute_error(y_test, predictions)

print("R2 Score:", r2)
print("Mean Absolute Error:", mae)

# VISUALISATION
import matplotlib.pyplot as plt

# Compare actual vs predicted
plt.figure(figsize=(8,5))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()
