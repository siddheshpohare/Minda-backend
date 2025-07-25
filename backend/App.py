from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import json
from datetime import datetime, timedelta
import traceback
import os
from werkzeug.utils import secure_filename

# --- App Initialization and Configuration ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes, allowing your React frontend to connect

# --- File Upload Settings ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Checks if the file's extension is in the ALLOWED_EXTENSIONS set."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Core Logic Class ---
class LSTMPredictor:
    """
    Manages data loading, model training, prediction, and metric calculation.
    This class holds the state of the application, including the current data file and the trained model.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.trained = False
        self.current_data_file = "T-13.xlsx"  # Default file on startup
        self.feature_columns = [] # Will be determined from the data
        self.seq_len = 10
        self.scaled_data = None

    def load_and_preprocess_data(self, file_path=None):
        """
        Loads data from the specified file, preprocesses it, and pivots it into the format required for the LSTM model.
        It dynamically identifies available feature columns from the file.
        """
        if file_path is None:
            file_path = self.current_data_file
            
        try:
            # Read Excel or CSV file
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            
            # --- FIX: Make data cleaning and type conversion robust to missing columns ---
            # Check for each column before trying to convert its type.
            if 'NonValueAddedCycleTime(Secs)' in df.columns:
                df["NonValueAddedCycleTime(Secs)"] = pd.to_numeric(df["NonValueAddedCycleTime(Secs)"], errors='coerce')
            if 'CycleTime(Secs)' in df.columns:
                df["CycleTime(Secs)"] = pd.to_numeric(df["CycleTime(Secs)"], errors='coerce')
            if 'AvgValue' in df.columns:
                df["AvgValue"] = pd.to_numeric(df["AvgValue"], errors='coerce')
            if 'StrokeStartTime' in df.columns:
                df["StrokeStartTime"] = pd.to_datetime(df["StrokeStartTime"], errors='coerce')
            
            # --- Map various process names to a standard set of feature names ---
            characteristic_mapping = {
                'Metal Temperature': 'metal_temperature', 'Metal Temperature  (Metal Temp)': 'metal_temperature', 'Metal Temp': 'metal_temperature',
                'Solidification Time': 'solidification_time', 'Solidification Time (Solid time)': 'solidification_time', 'Solid time': 'solidification_time',
                'Tilting Angle': 'tilting_angle', 'Tilting Angle (Tilting Angle)': 'tilting_angle',
                'Tilting Speed': 'tilting_speed', 'Tilting Speed in sec (Tilting Speed)': 'tilting_speed',
                'Top Die temperature': 'top_die_temperature', 'Top Die temperature (TDT)': 'top_die_temperature', 'TDT': 'top_die_temperature'
            }
            
            if 'ProcessCharacteristics' in df.columns:
                df['MappedCharacteristic'] = df['ProcessCharacteristics'].map(characteristic_mapping)
                relevant_characteristics = df.dropna(subset=['MappedCharacteristic'])
            else:
                return None, None # Not enough data to proceed

            if relevant_characteristics.empty:
                print("No relevant process characteristics found!")
                return None, None
            
            # --- Pivot the data to create a time-series format ---
            df_pivot = relevant_characteristics.pivot_table(
                index=['StrokeNumber', 'MachineName'], 
                columns='MappedCharacteristic', 
                values='AvgValue', 
                aggfunc='first'
            ).reset_index()
            
            # --- Dynamically set feature columns based on what's in the file ---
            default_features = ['metal_temperature', 'solidification_time', 'tilting_angle', 'tilting_speed', 'top_die_temperature']
            self.feature_columns = [col for col in default_features if col in df_pivot.columns]
            
            if not self.feature_columns:
                print("No feature columns found in pivoted data!")
                return None, None
            
            df_selected = df_pivot[['StrokeNumber', 'MachineName'] + self.feature_columns].dropna()
            
            if df_selected.empty:
                print("No data remaining after removing NaN values!")
                return None, None
            
            return df_selected, df

        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            traceback.print_exc()
            return None, None
    
    def create_sequences(self, data, seq_len):
        """Creates input sequences (X) and corresponding labels (y) for the LSTM model."""
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)
    
    def train_model(self, file_path=None):
        """
        Coordinates the full training pipeline: loading data, scaling, creating sequences, and fitting the LSTM model.
        """
        try:
            if file_path:
                self.current_data_file = file_path
                
            df_selected, _ = self.load_and_preprocess_data(self.current_data_file)
            if df_selected is None:
                return False, "Failed to load or preprocess data. Check file format and content."
            
            numeric_data = df_selected[self.feature_columns].astype(float)
            
            self.scaler = MinMaxScaler()
            self.scaled_data = self.scaler.fit_transform(numeric_data)
            
            self.seq_len = min(10, len(self.scaled_data) - 1)
            if self.seq_len < 2:
                return False, f"Insufficient data for training. Need at least {self.seq_len + 1} valid data points."
            
            X, y = self.create_sequences(self.scaled_data, self.seq_len)
            
            if len(X) == 0:
                return False, "Not enough data to create training sequences."
            
            self.model = Sequential([
                LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
                Dense(32, activation='relu'),
                Dense(len(self.feature_columns))
            ])
            self.model.compile(optimizer='adam', loss='mse')
            self.model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
            
            self.trained = True
            return True, f"Model trained successfully on {os.path.basename(self.current_data_file)} with {len(self.feature_columns)} features."
            
        except Exception as e:
            self.trained = False
            print(f"Training error: {e}")
            traceback.print_exc()
            return False, f"Training failed: {str(e)}"
    
    def predict_future(self, steps=20):
        """Generates future predictions using the trained model."""
        if not self.trained:
            return None, "Model has not been trained yet. Please upload a file and train the model."
        
        try:
            input_seq = self.scaled_data[-self.seq_len:].reshape(1, self.seq_len, len(self.feature_columns))
            future_predictions = []
            
            for _ in range(steps):
                pred = self.model.predict(input_seq, verbose=0)
                future_predictions.append(pred[0])
                input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, -1), axis=1)
            
            future_predictions = self.scaler.inverse_transform(np.array(future_predictions))
            
            now = datetime.now()
            timestamps = [(now + timedelta(minutes=i * 5)).strftime("%H:%M") for i in range(steps)]
            
            return future_predictions, timestamps
            
        except Exception as e:
            print(f"Prediction error: {e}")
            traceback.print_exc()
            return None, f"Prediction failed: {str(e)}"

    def calculate_metrics(self):
        """Calculates key performance indicators (KPIs) from the current data file."""
        try:
            # Use the pivoted data which is cleaner for this calculation
            df_selected, df_raw = self.load_and_preprocess_data(self.current_data_file)
            if df_selected is None:
                return None

            machines = df_selected["MachineName"].unique()
            metrics = {}

            # Define temperature thresholds
            thresholds = {
                'metal_temperature': {'min': 675, 'max': 755},
                'top_die_temperature': {'min': 270, 'max': 370},
            }

            for machine in machines:
                machine_data_pivoted = df_selected[df_selected["MachineName"] == machine]
                machine_data_raw = df_raw[df_raw["MachineName"] == machine]

                # --- Calculate Temperature Violations ---
                temp_violations = 0
                for param, limits in thresholds.items():
                    if param in machine_data_pivoted.columns:
                        # Count rows where the value is outside the min/max range
                        violations = machine_data_pivoted[
                            (machine_data_pivoted[param] < limits['min']) | 
                            (machine_data_pivoted[param] > limits['max'])
                        ]
                        temp_violations += len(violations)

                # --- Calculate Other Metrics (Robustly) ---
                idle_violations = 0
                if 'NonValueAddedCycleTime(Secs)' in machine_data_raw.columns:
                    idle_violations = machine_data_raw[machine_data_raw["NonValueAddedCycleTime(Secs)"] > 600]["StrokeNumber"].nunique()

                total_strokes = 0
                if 'StrokeNumber' in machine_data_raw.columns:
                    total_strokes = machine_data_raw["StrokeNumber"].nunique()

                utilization = 0
                if 'NonValueAddedCycleTime(Secs)' in machine_data_raw.columns and 'CycleTime(Secs)' in machine_data_raw.columns:
                    if not machine_data_raw.empty and machine_data_raw["CycleTime(Secs)"].sum() > 0:
                        utilization = (1 - machine_data_raw["NonValueAddedCycleTime(Secs)"].sum() / machine_data_raw["CycleTime(Secs)"].sum()) * 100

                metrics[machine] = {
                    "idle_time_violations": int(idle_violations),
                    "temperature_violations": int(temp_violations),  # Add the new metric
                    "total_strokes": int(total_strokes),
                    "machine_utilization": round(utilization, 2) if pd.notna(utilization) else 0
                }
            return metrics

        except Exception as e:
            print(f"Error calculating metrics: {e}")
            traceback.print_exc()
            return None

# Global instance of the predictor
predictor = LSTMPredictor()

# --- API Endpoints ---

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to confirm the server is running."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handles file uploads from the frontend."""
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"}), 400
        
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            predictor.current_data_file = filepath
            predictor.trained = False

            response_data = {
                "success": True,
                "message": f"File '{filename}' uploaded successfully.",
                "filename": filename
            }

            if request.form.get('auto_train', 'false').lower() == 'true':
                print(f"Auto-training model on {filename}...")
                success, train_message = predictor.train_model()
                response_data["auto_train_status"] = {"success": success, "message": train_message}
            
            return jsonify(response_data)

        except Exception as e:
            return jsonify({"success": False, "message": f"An error occurred: {str(e)}"}), 500
    else:
        return jsonify({"success": False, "message": "File type not allowed"}), 400

@app.route('/api/train', methods=['POST'])
def train_model_endpoint():
    """Endpoint to explicitly trigger model training."""
    success, message = predictor.train_model()
    if not success:
        return jsonify({"success": False, "message": message}), 400
    return jsonify({"success": True, "message": message, "feature_columns": predictor.feature_columns})

@app.route('/api/predict', methods=['GET'])
def get_predictions():
    """Returns future predictions based on the trained model."""
    steps = request.args.get('steps', 20, type=int)
    predictions, timestamps = predictor.predict_future(steps)
    
    if predictions is None:
        return jsonify({"success": False, "message": timestamps}), 400
    
    formatted_predictions = []
    for i, timestamp in enumerate(timestamps):
        p_dict = {"time": timestamp}
        for j, feature in enumerate(predictor.feature_columns):
            p_dict[feature] = round(float(predictions[i][j]), 2)
        formatted_predictions.append(p_dict)
        
    return jsonify({
        "success": True,
        "predictions": formatted_predictions,
        "feature_columns": predictor.feature_columns
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Calculates and returns KPIs for each machine."""
    metrics = predictor.calculate_metrics()
    if metrics is None:
        return jsonify({"success": False, "message": "Failed to calculate metrics. The file might be missing required columns."}), 400
    return jsonify({"success": True, "metrics": metrics})

@app.route('/api/machines', methods=['GET'])
def get_machines():
    """Extracts and returns a list of unique machine names."""
    try:
        _, df = predictor.load_and_preprocess_data()
        if df is None:
            return jsonify({"success": False, "message": "Could not load data to find machines. The file might be missing required columns."}), 400
            
        machines = df["MachineName"].unique().tolist()
        return jsonify({"success": True, "machines": machines})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error getting machines: {str(e)}"}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Generates real-time alerts by checking the latest data against thresholds."""
    try:
        df_selected, _ = predictor.load_and_preprocess_data()
        if df_selected is None:
            return jsonify({"success": False, "message": "Could not load data to generate alerts."}), 400
        
        alerts = []
        alert_id_counter = 1
        thresholds = {
            'metal_temperature': {'min': 675, 'max': 755},
            'top_die_temperature': {'min': 270, 'max': 370},
            'solidification_time': {'min': 30, 'max': 60}
        }
        
        latest_data_per_machine = df_selected.loc[df_selected.groupby('MachineName')['StrokeNumber'].idxmax()]
        
        for _, row in latest_data_per_machine.iterrows():
            machine_name = row['MachineName']
            for param, limits in thresholds.items():
                if param in row and pd.notna(row[param]):
                    value = row[param]
                    if value < limits['min'] or value > limits['max']:
                        alerts.append({
                            "id": alert_id_counter,
                            "machine": machine_name,
                            "parameter": param.replace('_', ' ').title(),
                            "value": round(value, 2),
                            "threshold": f"{limits['min']} - {limits['max']}",
                            "severity": "high" if value < limits['min'] * 0.95 or value > limits['max'] * 1.05 else "medium",
                            "time": datetime.now().strftime("%H:%M")
                        })
                        alert_id_counter += 1
        
        return jsonify({"success": True, "alerts": alerts})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Alerts error: {str(e)}"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    if os.path.exists(predictor.current_data_file):
        print(f"Default file '{predictor.current_data_file}' found. Attempting to train model on startup...")
        success, message = predictor.train_model()
        print(f"Startup Training Result: {message}")
    else:
        print(f"Default file '{predictor.current_data_file}' not found. Upload a file to train the model.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)