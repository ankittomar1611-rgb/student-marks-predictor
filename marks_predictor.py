import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [20, 30, 40, 50, 60, 70, 80, 90]
}

# DataFrame
df = pd.DataFrame(data)

print(df)

# Input and Output
X = df[["Hours"]]
y = df["Marks"]

# Model
model = LinearRegression()

# Train
model.fit(X, y)

# Prediction
hours = [[7.5]]

prediction = model.predict(hours)

print("\nPredicted Marks:", prediction[0])

# Graph
plt.scatter(df["Hours"], df["Marks"])

plt.plot(df["Hours"], model.predict(X))

plt.xlabel("Study Hours")
plt.ylabel("Marks")

plt.title("Student Marks Predictor")

plt.show()