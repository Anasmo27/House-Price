# =========================
# Predict House Price from Features (Area, Rooms, Bathrooms, Location) + Visualization
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -------------------------
# 1) Load dataset
# -------------------------
file_path = r"C:\Users\ASUS\Desktop\python test\House_Price\cairo_house_prices_expanded.csv"
data = pd.read_csv(file_path)

# Encode categorical column (Location)
le = LabelEncoder()
data["Location"] = le.fit_transform(data["Location"])

# Features (X) and Target (y)
X = data[["Area", "Rooms", "Bathrooms", "Location"]]
y = data["Price"]

# -------------------------
# 2) Split dataset
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 3) Models
# -------------------------
reg_models = {
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100)
}

for name, model in reg_models.items():
    model.fit(X_train, y_train)

# -------------------------
# 4) User Input
# -------------------------
print("\nEnter house details:")
user_area = float(input("Area (m²): "))
user_rooms = int(input("Rooms: "))
user_bathrooms = int(input("Bathrooms: "))
user_location = input("Location (e.g., Zamalek, Nasr City, Heliopolis): ")

# Convert location to encoded value
if user_location in le.classes_:
    user_location_encoded = le.transform([user_location])[0]
else:
    print("⚠️ Location not recognized, defaulting to 'Nasr City'")
    user_location_encoded = le.transform(["Nasr City"])[0]

user_input = np.array([[user_area, user_rooms, user_bathrooms, user_location_encoded]])

# -------------------------
# 5) Predictions for user input
# -------------------------
predictions = {}
print("\nPredicted house price (in EGP):")
for name, model in reg_models.items():
    pred_price = model.predict(user_input)[0]
    predictions[name] = pred_price
    print(f"{name}: {pred_price:,.0f} EGP")

# -------------------------
# 6) Evaluate models on test set
# -------------------------
print("\nModel evaluation on test data:")
for name, model in reg_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MSE={mse:.2f}, R²={r2:.2f}")

# -------------------------
# 7) Visualization
# -------------------------

# --- A) Bar chart for user input predictions ---
plt.figure(figsize=(6, 5))
plt.bar(predictions.keys(), predictions.values(), color=["orange", "blue", "green"])
plt.title(f"Predicted Price for Your Input\n(Area={user_area} m², Rooms={user_rooms}, "
          f"Baths={user_bathrooms}, Location={user_location})")
plt.ylabel("Price (EGP)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# --- B) Compare predictions vs actual on test data (random 6 samples) ---
sample = X_test.copy()
sample["Actual"] = y_test.values
for name, model in reg_models.items():
    sample[name] = model.predict(X_test)

sample_vis = sample.sample(6, random_state=42)

# Create subplots instead of multiple figures
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    row = sample_vis.iloc[i]
    actual = row["Actual"]
    preds = [row["KNN"], row["Decision Tree"], row["Random Forest"]]
    models = ["KNN", "Decision Tree", "Random Forest"]

    ax.bar(models, preds, color=["orange", "blue", "green"])
    ax.axhline(y=actual, color="red", linestyle="--", label=f"Actual: {actual:,.0f}")
    ax.set_title(f"Apt {i+1}: {row['Area']}m², {row['Rooms']}R, {row['Bathrooms']}B")
    ax.set_ylabel("Price (EGP)")
    ax.legend()

plt.tight_layout()
plt.show()

