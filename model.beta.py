# TO DO:
# 1. Clean up code; sort modules and functions sequentially.
# 2. Optimize and remove redundancies.
# 3. Improve precision and Accuracy

# libraries imported
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_curve,
    confusion_matrix,
)

linkedin_df_train = pd.read_csv("linkedin_train.csv")
linkedin_df_test = pd.read_csv("linkedin_test.csv")


# Performing Exploratory Data Analysis(EDA)
# Getting dataframe info
linkedin_df_train.info()
# Get the statistical summary of the dataframe
linkedin_df_train.describe()
# Checking if null values exist
linkedin_df_train.isnull().sum()
# Get the number of unique values in the "recommendation" feature
linkedin_df_train["No_Recommendation"].value_counts()


# Get the number of unique values in "real"
# This code needs to be fixed to see if real is being used while training.
# Real should only be used to check if answer is right or wrong.
linkedin_df_train["real"].value_counts()
linkedin_df_test.info()
linkedin_df_test.describe()
linkedin_df_test.isnull().sum()
linkedin_df_test["real"].value_counts()


# Visualize the data

sns.countplot(linkedin_df_train["real"])


plt.show()
# Visualize the private column data
sns.countplot(linkedin_df_train["No_Projects"])
plt.show()
# Visualize the "profile pic" column data
sns.countplot(linkedin_df_train["No_Professions"])
plt.show()
# Visualize the data
plt.figure(figsize=(20, 10))
sns.distplot(linkedin_df_train["No_Connections"])
plt.show()
# Correlation plot
plt.figure(figsize=(20, 20))
cm = linkedin_df_train.corr()
ax = plt.subplot()
# heatmap for correlation matrix
sns.heatmap(cm, annot=True, ax=ax)
plt.show()
sns.countplot(linkedin_df_test["No_Projects"])
sns.countplot(linkedin_df_test["No_Connections"])
sns.countplot(linkedin_df_test["No_Skills"])

# Preparing Data to Train the Model

Q1 = linkedin_df_train["No_Connections"].quantile(0.25)
Q3 = linkedin_df_train["No_Connections"].quantile(0.75)
IQR = Q3 - Q1
outliers = (linkedin_df_train["No_Connections"] < (Q1 - 1.5 * IQR)) | (
    linkedin_df_train["No_Connections"] > (Q3 + 1.5 * IQR)
)
linkedin_df_train = linkedin_df_train[~outliers]

# This part does not look good, check!
# Training and testing dataset (inputs)
X_train = linkedin_df_train.drop(columns=["real"])
X_test = linkedin_df_test.drop(columns=["real"])
# Training and testing dataset (Outputs)
y_train = linkedin_df_train["real"]
y_test = linkedin_df_test["real"]


# Scale the data before training the model
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)


# Build and Train Deep Learning Model
model = keras.Sequential(
    [
        Dense(50, input_dim=X_train.shape[1], activation="relu"),
        Dropout(0.5),
        Dense(150, activation="relu"),
        Dropout(0.3),
        Dense(150, activation="relu"),
        Dropout(0.3),
        Dense(100, activation="relu"),
        Dropout(0.3),
        Dense(25, activation="relu"),
        Dropout(0.3),
        Dense(2, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Split the training data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)
epochs_hist = model.fit(
    X_train_split, y_train_split, epochs=50, verbose=1, validation_data=(X_val, y_val)
)

# Access the Performance of the model
print("Training Accuracy:", max(epochs_hist.history["accuracy"]))
print("Validation Accuracy:", max(epochs_hist.history["val_accuracy"]))

# Display confusion matrix

# Lots of plotting.
plt.plot(epochs_hist.history["loss"])
plt.plot(epochs_hist.history["accuracy"])
plt.plot(epochs_hist.history["val_loss"])
plt.plot(epochs_hist.history["val_accuracy"])
plt.plot(epochs_hist.history["accuracy"])
plt.plot(epochs_hist.history["val_accuracy"])
plt.title("Model Accuracy Progression During Training/Validation")
plt.ylabel("Accuracy")
plt.xlabel("Epoch Number")
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.show()
plt.title("Model Loss Progression During Training/Validation")
plt.ylabel("Training and Validation Losses")
plt.xlabel("Epoch Number")
plt.legend(["Training Loss", "Validation Loss"])
plt.show()

predicted = model.predict(X_test)

predicted_value = []
test = []
for i in predicted:
    predicted_value.append(np.argmax(i))

for i in y_test:
    test.append(np.argmax(i))

print(classification_report(test, predicted_value))

plt.figure(figsize=(10, 10))
cm = confusion_matrix(test, predicted_value)
sns.heatmap(cm, annot=True)
plt.show()

# Temporary:

# Import necessary libraries
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score

# 1. Handle Class Imbalance
class_weights = class_weight.compute_class_weight(
    "balanced", classes=np.unique(y_train[:, 0]), y=y_train[:, 0].tolist()
)
class_weights_dict = dict(enumerate(class_weights))

# 8. Evaluate on Multiple Metrics
y_pred = model.predict(X_test_scaled)
y_pred_class = np.argmax(y_pred, axis=1)

precision = precision_score(test, y_pred_class, average="binary", zero_division=1)
recall = recall_score(test, y_pred_class, average="binary", zero_division=1)
f1 = f1_score(test, y_pred_class, average="binary", zero_division=1)

print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

# 9. Address UndefinedMetricWarning
print(classification_report(test, y_pred_class, zero_division=1))


from tensorflow.keras.models import save_model

# Save the model
model.save("your_model.h5")

# from joblib import dump

# Replace 'your_model' with 'model'
# dump(model, 'your_model.h5')
