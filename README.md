🏠 House Price Prediction in Cairo

This project predicts house prices in Cairo based on Area, Rooms, Bathrooms, and Location using multiple machine learning models.
It also provides visualizations to compare predictions across models and against actual data.

📌 Features

Train three regression models:

K-Nearest Neighbors (KNN)

Decision Tree Regressor

Random Forest Regressor

Encode categorical feature (Location) automatically.

Accept user input (Area, Rooms, Bathrooms, Location) to predict house price.

Evaluate models using MSE and R² score.

Visualize predictions:

📊 Bar chart comparing predicted prices from all models for user input.

📉 Subplots comparing predictions vs. actual prices for test samples.

📂 Dataset

The dataset used is:

cairo_house_prices_expanded.csv


It includes:

Area (m²)

Rooms (integer)

Bathrooms (integer)

Location (categorical, e.g., Zamalek, Nasr City, Heliopolis)

Price (EGP)

⚙️ Installation

Clone this repository:

git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction


Install dependencies:

pip install -r requirements.txt


Place the dataset in the project folder:

cairo_house_prices_expanded.csv

🚀 Usage

Run the Python script:

python house_price_prediction.py


You will be prompted to enter details:

Enter house details:
Area (m²): 150
Rooms: 3
Bathrooms: 2
Location (e.g., Zamalek, Nasr City, Heliopolis): Nasr City


The program will:

Print predicted prices from each model.

Show evaluation metrics (MSE, R²).

Display visualizations with results.

📊 Example Visualization

Bar chart for user input predictions:

Comparison plots (Predictions vs Actual for random test samples):

📈 Model Evaluation

For each model, the program prints:

KNN: MSE=123456.78, R²=0.85
Decision Tree: MSE=98765.43, R²=0.88
Random Forest: MSE=54321.10, R²=0.92

🛠️ Requirements

See requirements.txt:

numpy
pandas
matplotlib
scikit-learn

📌 Future Improvements

Add more features (Year Built, Furnishing, Floor, etc.).

Deploy as a web app using Streamlit or Flask.

Try advanced models like XGBoost or Neural Networks.