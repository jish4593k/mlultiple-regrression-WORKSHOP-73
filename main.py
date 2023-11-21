import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
datasets = pd.read_csv('50_Startups.csv')
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 4].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

label_encoder = LabelEncoder()
X[:, 3] = label_encoder.fit_transform(X[:, 3])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
    remainder='passthrough'
)
X = ct.fit_transform(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Build the model using TensorFlow and Keras
model = keras.Sequential([
    layers.Dense(units=1, input_dim=X_Train.shape[1])
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_Train, Y_Train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Predicting the Test set results
Y_Pred = model.predict(X_Test)

# Evaluate the model
mse = mean_squared_error(Y_Test, Y_Pred)
print(f'Mean Squared Error on Test Set: {mse}')

# Visualizing the training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualizing the results
plt.scatter(Y_Test, Y_Pred, color='blue')
plt.plot([min(Y_Test), max(Y_Test)], [min(Y_Test), max(Y_Test)], linestyle='--', color='red')
plt.title('Actual vs Predicted (Test set)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# Visualizing the residuals
residuals = Y_Test - Y_Pred.flatten()
plt.scatter(Y_Pred.flatten(), residuals, color='blue')
plt.hlines(y=0, xmin=min(Y_Pred.flatten()), xmax=max(Y_Pred.flatten()), linestyle='--', color='red')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()
