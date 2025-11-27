import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import shap
import joblib
import os

# --- CONFIGURATION ---
SEQUENCE_LENGTH = 50  # Sliding window size
FEAT_COLS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
MODEL_PATH = "lstm_rul_model.h5"
SCALER_PATH = "scaler.pkl"

def generate_mock_nasa_data(num_units=100):
    """
    Generates synthetic data mimicking NASA C-MAPSS structure.
    Real data would be loaded from .txt files.
    """
    print("Generating synthetic NASA Turbofan data...")
    data_list = []
    
    for unit in range(1, num_units + 1):
        # Random life for each engine (150 to 300 cycles)
        max_life = np.random.randint(150, 300)
        
        for cycle in range(1, max_life + 1):
            # Linearly degrading health factor (1.0 to 0.0)
            health = 1.0 - (cycle / max_life)
            
            # Simulate sensors (adding noise to the health factor)
            # Some sensors go up as health degrades, some go down
            row = {'unit': unit, 'cycle': cycle}
            
            # Add dummy settings
            row['setting1'] = np.random.normal(0, 1)
            
            # Generate sensor data based on health + noise
            for i, sensor in enumerate(FEAT_COLS):
                noise = np.random.normal(0, 0.05)
                # Arbitrary logic: even index sensors rise, odd fall
                if i % 2 == 0:
                    val = (1 - health) * 100 + noise # Rising trend
                else:
                    val = health * 100 + noise # Falling trend
                row[sensor] = val
            
            data_list.append(row)
            
    df = pd.DataFrame(data_list)
    print(f"Data generated. Shape: {df.shape}")
    return df

def process_data(df):
    """
    1. Calculate RUL (Remaining Useful Life)
    2. Scale Data
    3. Create Sliding Windows
    """
    # 1. Calculate RUL
    # RUL = Max Cycle - Current Cycle
    max_cycles = df.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max']
    df = df.merge(max_cycles, on='unit', how='left')
    df['RUL'] = df['max'] - df['cycle']
    df.drop('max', axis=1, inplace=True)
    
    # 2. Scale Features
    scaler = MinMaxScaler()
    df[FEAT_COLS] = scaler.fit_transform(df[FEAT_COLS])
    
    print("Data scaled and RUL calculated.")
    return df, scaler

def create_sequences(df, seq_length, feature_cols):
    """
    Converts DataFrame into 3D array (Samples, TimeSteps, Features) for LSTM
    """
    X = []
    y = []
    
    for unit in df['unit'].unique():
        unit_data = df[df['unit'] == unit]
        
        # Need at least seq_length data points
        if len(unit_data) < seq_length:
            continue
            
        data_array = unit_data[feature_cols].values
        rul_array = unit_data['RUL'].values
        
        for i in range(len(data_array) - seq_length):
            X.append(data_array[i : i + seq_length])
            y.append(rul_array[i + seq_length])
            
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='linear') # Regression output
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def explain_model(model, X_train, feature_names):
    """
    Generates SHAP values for a subset of data
    """
    print("\n--- SHAP Explainability ---")
    # We use a small background sample for DeepExplainer to keep it fast
    background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
    
    # Note: SHAP DeepExplainer compatibility varies with TF versions.
    # We wrap in try-except for robust demo execution.
    try:
        explainer = shap.DeepExplainer(model, background)
        # Explain 5 samples
        shap_values = explainer.shap_values(X_train[:5])
        print("SHAP values generated successfully.")
        print(f"Shape of SHAP values: {np.array(shap_values).shape}")
        # In a real notebook, you would plot this: shap.summary_plot(shap_values, ...)
    except Exception as e:
        print(f"SHAP generation skipped (version compatibility check): {e}")

def main():
    # 1. Load Data
    df = generate_mock_nasa_data()
    
    # 2. Preprocess
    df_processed, scaler = process_data(df)
    
    # 3. Create Sequences
    X, y = create_sequences(df_processed, SEQUENCE_LENGTH, FEAT_COLS)
    print(f"Input Shape: {X.shape}, Target Shape: {y.shape}")
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Build & Train
    model = build_model((SEQUENCE_LENGTH, len(FEAT_COLS)))
    print("\nTraining LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=5, # Low epochs for demo speed; use 50+ for real results
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # 6. Evaluate
    loss, mae = model.evaluate(X_test, y_test)
    print(f"\nTest MAE: {mae:.2f} cycles")
    
    # 7. Explain
    explain_model(model, X_train, FEAT_COLS)
    
    # 8. Save Artifacts for API
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

if __name__ == "__main__":
    main()