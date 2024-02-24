# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, confusion_matrix
from sklearn.utils import class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

# Loading data
linkedin_df_train = pd.read_csv("linkedin_train.csv")
linkedin_df_test = pd.read_csv("linkedin_test.csv")

# Exploratory Data Analysis (EDA)
linkedin_df_train.info()
linkedin_df_train.describe()
linkedin_df_train.isnull().sum()
linkedin_df_train["No_Recommendation"].value_counts()

linkedin_df_train["real"].value_counts()
linkedin_df_test.info()
linkedin_df_test.describe()
linkedin_df_test.isnull().sum()
linkedin_df_test["real"].value_counts()

# Data Visualization
sns.countplot(linkedin_df_train["real"])
plt.show()
sns.countplot(linkedin_df_train["No_Projects"])
plt.show()
sns.countplot(linkedin_df_train["No_Professions"])
plt.show()
plt.figure(figsize=(20, 10))
sns.distplot(linkedin_df_train["No_Connections"])
plt.show()
plt.figure(figsize=(20, 20))
cm = linkedin_df_train.corr()
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax)
plt.show()
sns.countplot(linkedin_df_test["No_Projects"])
sns.countplot(linkedin_df_test["No_Connections"])
sns.countplot(linkedin_df_test["No_Skills"])

# Preparing Data
Q1 = linkedin_df_train["No_Connections"].quantile(0.25)
Q3 = linkedin_df_train["No_Connections"].quantile(0.75)
IQR = Q3 - Q1
outliers = (linkedin_df_train["No_Connections"] < (Q1 - 1.5 * IQR)) | (linkedin_df_train["No_Connections"] > (Q3 + 1.5 * IQR))
linkedin_df_train = linkedin_df_train[~outliers]

X_train = linkedin_df_train.drop(columns=["real"])
X_test = linkedin_df_test.drop(columns=["real"])
y_train = linkedin_df_train["real"]
y_test = linkedin_df_test["real"]

# Scaling data
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# Model Architecture
model = keras.Sequential([
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
    Dense(2, activation="softmax")
])

# Compiling the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Splitting data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Training the model
epochs_hist = model.fit(X_train_split, y_train_split, epochs=50, verbose=1, validation_data=(X_val, y_val))

# Model Evaluation
print("Training Accuracy:", max(epochs_hist.history["accuracy"]))
print("Validation Accuracy:", max(epochs_hist.history["val_accuracy"]))

# Confusion Matrix
plt.plot(epochs_hist.history["loss"])
plt.plot(epochs_hist.history["accuracy"])
plt.plot(epochs_hist.history["val_loss"])
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

# Predictions
predicted = model.predict(X_test)
predicted_value = [np.argmax(i) for i in predicted]
test = [np.argmax(i) for i in y_test]
print(classification_report(test, predicted_value))

# Confusion Matrix
plt.figure(figsize=(10, 10))
cm = confusion_matrix(test, predicted_value)
sns.heatmap(cm, annot=True)
plt.show()

# Handling Class Imbalance
class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train[:, 0]), y=y_train[:, 0].tolist())
class_weights_dict = dict(enumerate(class_weights))

# Evaluating on Multiple Metrics
y_pred = model.predict(X_test_scaled)
y_pred_class = np.argmax(y_pred, axis=1)
precision = precision_score(test, y_pred_class, average="binary", zero_division=1)
recall = recall_score(test, y_pred_class, average="binary", zero_division=1)
f1 = f1_score(test, y_pred_class, average="binary", zero_division=1)
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

# Classification Report with no zero_division warning
print(classification_report(test, y_pred_class, zero_division=1))

# Saving the model
model.save("your_model.h5")
