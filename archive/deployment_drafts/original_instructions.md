Predictive Maintenance System (LSTM + FastAPI)

An end-to-end MLOps project for forecasting Remaining Useful Life (RUL) of turbofan engines using Deep Learning.

Project Architecture

Data: Simulates NASA C-MAPSS data (Multivariate Time Series).

Model: LSTM (Long Short-Term Memory) Neural Network.

Explainability: SHAP (DeepExplainer).

Deployment: FastAPI inside Docker.

How to Run Locally

1. Install Dependencies

pip install -r requirements.txt


2. Train the Model

This will generate synthetic data, train the LSTM, and save lstm_rul_model.h5 and scaler.pkl.

python train_pipeline.py


3. Start the API

uvicorn app:app --reload


Open your browser to http://127.0.0.1:8000/docs to see the Swagger UI.

How to Run with Docker

Train the model first (ensure .h5 and .pkl files are in the folder).

Build the image:

docker build -t pred-maint-app .


Run the container:

docker run -p 8000:8000 pred-maint-app


API Usage Example

Send a POST request to /predict with a JSON payload containing the last 50 cycles of sensor data for a specific unit.