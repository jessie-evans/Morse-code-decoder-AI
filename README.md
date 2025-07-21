***Morse Code Tap Classifier Morse Code Tap Classifier***
This project contains a Python script to train a machine learning model that can classify user tap inputs as either a "dot" or a "dash" in Morse code. The classification is based on the duration and velocity of the taps.

The script processes pre-collected tap data, trains a RandomForestClassifier, evaluates its performance, and saves the trained model to a file (tap_classifier_model.pkl) for use in a real-time application.

📋 **Table of Contents**
How It Works

Dataset Requirements

Dependencies

How to Run

Script Output

Next Steps

🧠 ***How It Works***
The train_model.py script follows these key steps:

**Load Data:** It reads two separate CSV files: dot_data.csv containing data for short taps (dots) and dash_data.csv for long taps (dashes).

**Labeling:** It assigns a numerical label to each dataset: 0 for dots and 1 for dashes.

**Combine and Shuffle:** The two datasets are concatenated into a single DataFrame. This combined dataset is then shuffled randomly to ensure the model doesn't learn from the order of the data.

**Feature Selection:** The duration and velocity columns are selected as the features (X), and the label column is used as the target (y).

**Train-Test Split:** The data is split into a training set (80%) and a testing set (20%) to evaluate the model's performance on unseen data.

**Model Training:** A RandomForestClassifier is initialized and trained on the training data (X_train, y_train).

**Evaluation:** The trained model makes predictions on the test set. A Confusion Matrix and a Classification Report (including precision, recall, and F1-score) are printed to the console to assess the model's accuracy.

**Save Model:** The trained model object is serialized and saved to a file named tap_classifier_model.pkl using joblib. This file can then be loaded into another application to make real-time predictions.

📊 ***Dataset Requirements***
For the script to run correctly, you must have two CSV files in the same directory:

**dot_data.csv:** Contains data for taps intended to be "dots".

**dash_data.csv:** Contains data for taps intended to be "dashes".

Both files must have the following columns:

**duration:** The time the tap was held down (e.g., in milliseconds).

**velocity:** The speed or force of the tap (if available from the input device).

Example dot_data.csv:

duration,velocity
110,0.8
95,0.9
125,0.75

⚙️ ***Dependencies***
This project requires Python 3 and the following libraries. You can install them using pip:

pip install pandas scikit-learn joblib

**Pandas:** For data manipulation and reading CSV files.

**Scikit-learn:** For machine learning (train-test split, model, and metrics).

**Joblib:** For efficiently saving and loading the trained model.

▶️ ***How to Run***
Ensure you have your dot_data.csv and dash_data.csv files ready.

Save the code as a Python file (e.g., train_model.py).

Run the script from your terminal:

python train_model.py

📈 ***Script Output***
After running the script, you will see the following in your console:

**Confusion Matrix:** Shows the number of correct and incorrect predictions for each class (dot/dash).

**Classification Report:** Provides detailed metrics like precision, recall, and f1-score for each class.

**Success Message:** A confirmation that the model has been saved.

**Confusion Matrix:**
[[...]]

**Classification Report:**
              precision    recall  f1-score   support
           0       ...       ...       ...       ...
           1       ...       ...       ...       ...
    ...

✅ Model saved as tap_classifier_model.pkl

A new file, tap_classifier_model.pkl, will be created in your project directory.

🚀 Next Steps
The saved tap_classifier_model.pkl file is ready to be integrated into a real-time Morse code application. You can load this model in another script to predict whether a live user tap is a dot or a dash based on its measured duration and velocity.

Example of loading the model:

import joblib

# Load the trained model
model = joblib.load("tap_classifier_model.pkl")

# Predict a new tap with duration=100ms and velocity=0.85
new_tap_data = [[100, 0.85]]
prediction = model.predict(new_tap_data)

if prediction[0] == 0:
    print("Predicted: Dot")
else:
    print("Predicted: Dash")
