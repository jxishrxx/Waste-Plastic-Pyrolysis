import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from keras.models import Sequential
from keras.layers import LSTM, Dense

dataset = pd.read_csv('plastic_data.csv')

X = dataset.drop(columns=['solid_yield', 'liquid_yield', 'gas_yield', 'aromatic_yield', 'styrene_yield'])
y = dataset[['solid_yield', 'liquid_yield', 'gas_yield', 'aromatic_yield', 'styrene_yield']]

# Scale input features between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape features for input to the RNN model
X_train_rnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the RNN model
model_rnn = Sequential()
model_rnn.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_rnn.shape[1], 1)))
model_rnn.add(LSTM(units=50))
model_rnn.add(Dense(units=5))  # Output layer with 5 neurons for multi-output regression
model_rnn.compile(optimizer='adam', loss='mean_squared_error')
model_rnn.fit(X_train_rnn, y_train, epochs=100, batch_size=32)

# Build the SVM model
model_svm = SVR(kernel='rbf')
model_svm.fit(X_train, y_train)

# Build the GP model
kernel = RBF(length_scale=1.0)
model_gp = GaussianProcessRegressor(kernel=kernel)
model_gp.fit(X_train, y_train)

# Evaluate RNN model
X_test_rnn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
rnn_predictions = model_rnn.predict(X_test_rnn)

# Evaluate SVM model
svm_predictions = model_svm.predict(X_test)

# Evaluate GP model
gp_predictions = model_gp.predict(X_test)

# Calculate R^2 scores for all models
rnn_score = model_rnn.score(X_test_rnn, y_test)
svm_score = model_svm.score(X_test, y_test)
gp_score = model_gp.score(X_test, y_test)

print("RNN R^2 Score:", rnn_score)
print("SVM R^2 Score:", svm_score)
print("GP R^2 Score:", gp_score)

# Make predictions for a specific plastic sample
input_sample = scaler.transform([[plastic_type, carbon_content, hydrogen_content, particle_size]])

rnn_prediction = model_rnn.predict(np.reshape(input_sample, (1, -1, 1)))
svm_prediction = model_svm.predict(input_sample)
gp_prediction = model_gp.predict(input_sample)

print("RNN Prediction:", rnn_prediction)
print("SVM Prediction:", svm_prediction)
print("GP Prediction:", gp_prediction)


from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    X_input = np.array(data['features'])
    X_input_scaled = scaler.transform(X_input)
    X_input_reshaped = np.reshape(X_input_scaled, (X_input.shape[0], X_input_scaled.shape[1], 1))

    predictions = model.predict(X_input_reshaped)

    return jsonify({'predictions': predictions.tolist()})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
