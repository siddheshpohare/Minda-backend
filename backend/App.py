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

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class LSTMPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'metal_temperature',
            'solidification_time', 
            'tilting_angle',
            'tilting_speed',
            'top_die_temperature'
        ]
        self.trained = False
        self.current_data_file = "T-13.xlsx"  # Default data file
        
    def load_and_preprocess_data(self, file_path=None):
        """Load and preprocess data from Excel file"""
        if file_path is None:
            file_path = self.current_data_file
            
        try:
            # Read Excel or CSV file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Print column info for debugging
            print("Available columns:", df.columns.tolist())
            print("Data shape:", df.shape)
            
            # Data type conversions
            df["NonValueAddedCycleTime(Secs)"] = pd.to_numeric(df["NonValueAddedCycleTime(Secs)"], errors='coerce')
            df["CycleTime(Secs)"] = pd.to_numeric(df["CycleTime(Secs)"], errors='coerce')
            df["AvgValue"] = pd.to_numeric(df["AvgValue"], errors='coerce')
            df["MinValue"] = pd.to_numeric(df["MinValue"], errors='coerce')
            df["MaxValue"] = pd.to_numeric(df["MaxValue"], errors='coerce')
            df["StrokeStartTime"] = pd.to_datetime(df["StrokeStartTime"], errors='coerce')
            df["StepStartTime"] = pd.to_datetime(df["StepStartTime"], errors='coerce')
            
            # Print unique ProcessCharacteristics for debugging
            print("Available ProcessCharacteristics:")
            print(df["ProcessCharacteristics"].unique())
            
            # Map process characteristics to standard names
            characteristic_mapping = {
                # Add your actual process characteristic names here
                'Metal Temperature': 'metal_temperature',
                'Metal Temperature  (Metal Temp)': 'metal_temperature',
                'Metal Temp': 'metal_temperature',
                'Solidification Time': 'solidification_time',
                'Solidification Time (Solid time)': 'solidification_time',
                'Solid time': 'solidification_time',
                'Tilting Angle': 'tilting_angle',
                'Tilting Angle (Tilting Angle)': 'tilting_angle',
                'Tilting Speed': 'tilting_speed',
                'Tilting Speed in sec (Tilting Speed)': 'tilting_speed',
                'Top Die temperature': 'top_die_temperature',
                'Top Die temperature (TDT)': 'top_die_temperature',
                'TDT': 'top_die_temperature'
            }
            
            # Filter relevant process characteristics
            relevant_characteristics = df[df["ProcessCharacteristics"].isin(characteristic_mapping.keys())]
            
            if relevant_characteristics.empty:
                print("No relevant process characteristics found!")
                return None, None
            
            # Create mapped characteristic names
            relevant_characteristics = relevant_characteristics.copy()
            relevant_characteristics['MappedCharacteristic'] = relevant_characteristics['ProcessCharacteristics'].map(characteristic_mapping)
            
            # Pivot data for LSTM input
            df_pivot = relevant_characteristics.pivot_table(
                index=['StrokeNumber', 'MachineName'], 
                columns='MappedCharacteristic', 
                values='AvgValue', 
                aggfunc='first'
            )
            
            # Reset index to make StrokeNumber and MachineName regular columns
            df_pivot = df_pivot.reset_index()
            
            # Select only the feature columns that exist
            existing_features = [col for col in self.feature_columns if col in df_pivot.columns]
            
            if not existing_features:
                print("No feature columns found in pivoted data!")
                return None, None
            
            # Update feature columns to only include existing ones
            self.feature_columns = existing_features
            df_selected = df_pivot[['StrokeNumber', 'MachineName'] + self.feature_columns]
            
            # Remove rows with NaN values
            df_selected = df_selected.dropna()
            
            if df_selected.empty:
                print("No data remaining after removing NaN values!")
                return None, None
            
            print(f"Processed data shape: {df_selected.shape}")
            print(f"Feature columns: {self.feature_columns}")
            
            return df_selected, df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return None, None
    
    def create_sequences(self, data, seq_len):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)
    
    def train_model(self, file_path=None):
        """Train the LSTM model"""
        try:
            if file_path:
                self.current_data_file = file_path
                
            df_selected, df = self.load_and_preprocess_data(file_path)
            if df_selected is None:
                return False, "Failed to load data"
            
            # Use only numeric feature columns for training
            numeric_data = df_selected[self.feature_columns].astype(float)
            
            # Normalize data
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(numeric_data)
            
            # Create sequences
            SEQ_LEN = min(10, len(scaled_data) - 1)  # Adjust sequence length based on data size
            if SEQ_LEN < 2:
                return False, "Insufficient data for training (need at least 3 samples)"
            
            X, y = self.create_sequences(scaled_data, SEQ_LEN)
            
            if len(X) == 0:
                return False, "No training sequences could be created"
            
            X = X.reshape((X.shape[0], SEQ_LEN, X.shape[2]))
            
            # Build and train model
            self.model = Sequential([
                LSTM(64, input_shape=(SEQ_LEN, len(self.feature_columns)), return_sequences=False),
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(len(self.feature_columns))
            ])
            
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train with validation split
            validation_split = 0.2 if len(X) > 10 else 0.1
            history = self.model.fit(
                X, y, 
                epochs=50, 
                batch_size=min(16, len(X)//2), 
                validation_split=validation_split, 
                verbose=1
            )
            
            self.trained = True
            self.scaled_data = scaled_data
            self.seq_len = SEQ_LEN
            
            return True, f"Model trained successfully with {len(X)} samples"
            
        except Exception as e:
            print(f"Training error: {e}")
            traceback.print_exc()
            return False, f"Training failed: {str(e)}"
    
    def predict_future(self, steps=20):
        """Generate future predictions"""
        if not self.trained:
            return None, "Model not trained"
        
        try:
            # Use the last sequence from training data
            input_seq = self.scaled_data[-self.seq_len:].reshape(1, self.seq_len, len(self.feature_columns))
            future_predictions = []
            
            for _ in range(steps):
                pred = self.model.predict(input_seq, verbose=0)
                future_predictions.append(pred[0])
                # Update input sequence for next prediction
                input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, -1), axis=1)
            
            # Inverse transform predictions
            future_predictions = self.scaler.inverse_transform(np.array(future_predictions))
            
            # Generate timestamps
            current_time = datetime.now()
            timestamps = [(current_time + timedelta(minutes=i*5)).strftime("%H:%M") for i in range(steps)]
            
            return future_predictions, timestamps
            
        except Exception as e:
            print(f"Prediction error: {e}")
            traceback.print_exc()
            return None, f"Prediction failed: {str(e)}"
    
    def calculate_metrics(self, file_path=None):
        """Calculate production metrics"""
        try:
            df_selected, df = self.load_and_preprocess_data(file_path)
            if df is None:
                return None
            
            # Get unique machines
            machines = df["MachineName"].unique()
            metrics = {}
            
            for machine in machines:
                machine_data = df[df["MachineName"] == machine]
                
                # Filter for recent data (last day)
                if not machine_data["StrokeStartTime"].empty:
                    latest_date = machine_data["StrokeStartTime"].max().date()
                    machine_data = machine_data[machine_data["StrokeStartTime"].dt.date == latest_date]
                
                # A. Idle Time > 10 min (600 seconds)
                idle_violations = machine_data[machine_data["NonValueAddedCycleTime(Secs)"] > 600]["StrokeNumber"].nunique()
                
                # B. Temperature violations (example thresholds)
                temp_violations = 0
                metal_temp_data = machine_data[machine_data["ProcessCharacteristics"].str.contains("Metal Temperature", case=False, na=False)]
                if not metal_temp_data.empty:
                    temp_violations += len(metal_temp_data[(metal_temp_data["AvgValue"] < 675) | (metal_temp_data["AvgValue"] > 755)])
                
                die_temp_data = machine_data[machine_data["ProcessCharacteristics"].str.contains("Top Die", case=False, na=False)]
                if not die_temp_data.empty:
                    temp_violations += len(die_temp_data[(die_temp_data["AvgValue"] < 270) | (die_temp_data["AvgValue"] > 370)])
                
                # C. Quality metrics
                total_strokes = machine_data["StrokeNumber"].nunique()
                avg_cycle_time = machine_data["CycleTime(Secs)"].mean()
                
                metrics[machine] = {
                    "idle_time_violations": idle_violations,
                    "temperature_violations": temp_violations,
                    "total_strokes": total_strokes,
                    "avg_cycle_time": round(avg_cycle_time, 2) if not pd.isna(avg_cycle_time) else 0,
                    "machine_utilization": round((1 - machine_data["NonValueAddedCycleTime(Secs)"].mean() / machine_data["CycleTime(Secs)"].mean()) * 100, 2) if not machine_data.empty else 0
                }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            traceback.print_exc()
            return None

# Initialize predictor
predictor = LSTMPredictor()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_trained": predictor.trained,
        "feature_columns": predictor.feature_columns,
        "current_data_file": predictor.current_data_file,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload data file endpoint"""
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "message": "No file provided",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                "success": False,
                "message": "No file selected",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "message": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Secure the filename and save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        # Validate file by trying to read it
        try:
            if filename.endswith('.csv'):
                test_df = pd.read_csv(filepath)
            else:
                test_df = pd.read_excel(filepath)
            
            # Check if required columns exist
            required_columns = ['MachineName', 'ProcessCharacteristics', 'AvgValue', 'StrokeNumber']
            missing_columns = [col for col in required_columns if col not in test_df.columns]
            
            if missing_columns:
                # Clean up uploaded file
                os.remove(filepath)
                return jsonify({
                    "success": False,
                    "message": f"Missing required columns: {', '.join(missing_columns)}",
                    "timestamp": datetime.now().isoformat()
                }), 400
            
            # Get auto-train option from request
            auto_train = request.form.get('auto_train', 'false').lower() == 'true'
            
            response_data = {
                "success": True,
                "message": "File uploaded successfully",
                "filename": filename,
                "filepath": filepath,
                "rows": len(test_df),
                "columns": len(test_df.columns),
                "machines": test_df['MachineName'].nunique() if 'MachineName' in test_df.columns else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Auto-train model if requested
            if auto_train:
                success, train_message = predictor.train_model(filepath)
                response_data["auto_train"] = {
                    "success": success,
                    "message": train_message
                }
                if success:
                    response_data["message"] += " and model trained successfully"
            
            return jsonify(response_data)
            
        except Exception as e:
            # Clean up uploaded file if validation fails
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                "success": False,
                "message": f"Invalid file format: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Upload error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/files', methods=['GET'])
def list_files():
    """List uploaded files"""
    try:
        files = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if allowed_file(filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file_stats = os.stat(filepath)
                
                files.append({
                    "filename": filename,
                    "size": file_stats.st_size,
                    "uploaded": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    "is_current": filepath == predictor.current_data_file
                })
        
        return jsonify({
            "success": True,
            "files": files,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error listing files: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/switch-file', methods=['POST'])
def switch_file():
    """Switch to a different uploaded file"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({
                "success": False,
                "message": "Filename not provided",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                "success": False,
                "message": "File not found",
                "timestamp": datetime.now().isoformat()
            }), 404
        
        # Switch to new file
        predictor.current_data_file = filepath
        
        # Optionally retrain model
        retrain = data.get('retrain', False)
        if retrain:
            success, message = predictor.train_model(filepath)
            return jsonify({
                "success": True,
                "message": f"Switched to {filename}. {message}",
                "retrain_success": success,
                "timestamp": datetime.now().isoformat()
            })
        
        return jsonify({
            "success": True,
            "message": f"Switched to {filename}",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Switch file error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the LSTM model"""
    try:
        # Get optional file parameter
        data = request.get_json() if request.is_json else {}
        file_path = data.get('file_path') if data else None

        success, message = predictor.train_model(file_path)
        if not success:
            return jsonify({
                "success": False,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }), 400  # Return 400 for training errors

        return jsonify({
            "success": success,
            "message": message,
            "feature_columns": predictor.feature_columns,
            "data_file": predictor.current_data_file,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Training error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Training error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/predict', methods=['GET'])
def get_predictions():
    """Get LSTM predictions for dashboard"""
    try:
        steps = request.args.get('steps', 20, type=int)
        predictions, timestamps = predictor.predict_future(steps)
        
        if predictions is None:
            return jsonify({
                "success": False,
                "message": timestamps,  # Error message
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Format predictions for frontend
        formatted_predictions = []
        for i, timestamp in enumerate(timestamps):
            prediction_dict = {"time": timestamp}
            
            # Map predictions to available features
            for j, feature in enumerate(predictor.feature_columns):
                if j < len(predictions[i]):
                    prediction_dict[feature] = float(predictions[i][j])
            
            formatted_predictions.append(prediction_dict)
        
        return jsonify({
            "success": True,
            "predictions": formatted_predictions,
            "feature_columns": predictor.feature_columns,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Prediction error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get production metrics"""
    try:
        metrics = predictor.calculate_metrics()
        
        if metrics is None:
            return jsonify({
                "success": False,
                "message": "Failed to calculate metrics",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        return jsonify({
            "success": True,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Metrics error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/machines', methods=['GET'])
def get_machines():
    """Get list of available machines"""
    try:
        # Read from current data file
        if predictor.current_data_file.endswith('.csv'):
            df = pd.read_csv(predictor.current_data_file)
        else:
            df = pd.read_excel(predictor.current_data_file)
            
        machines = df["MachineName"].unique().tolist()
        
        return jsonify({
            "success": True,
            "machines": machines,
            "data_file": predictor.current_data_file,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error getting machines: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/current-data', methods=['GET'])
def get_current_data():
    """Get current machine data"""
    try:
        # Read from current data file
        if predictor.current_data_file.endswith('.csv'):
            df = pd.read_csv(predictor.current_data_file)
        else:
            df = pd.read_excel(predictor.current_data_file)
            
        machines = df["MachineName"].unique()
        
        current_data = {}
        
        for machine in machines:
            # Get latest data for each machine
            machine_data = df[df["MachineName"] == machine]
            latest_stroke = machine_data["StrokeNumber"].max()
            latest_data = machine_data[machine_data["StrokeNumber"] == latest_stroke]
            
            machine_readings = {}
            for _, row in latest_data.iterrows():
                characteristic = row["ProcessCharacteristics"]
                if pd.notna(row["AvgValue"]):
                    machine_readings[characteristic] = float(row["AvgValue"])
            
            current_data[machine] = machine_readings
        
        return jsonify({
            "success": True,
            "data": current_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Data error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get current alerts based on actual data"""
    try:
        # Read from current data file
        if predictor.current_data_file.endswith('.csv'):
            df = pd.read_csv(predictor.current_data_file)
        else:
            df = pd.read_excel(predictor.current_data_file)
            
        alerts = []
        alert_id = 1
        
        # Define thresholds for different parameters
        thresholds = {
            'Metal Temperature': {'min': 675, 'max': 755},
            'Top Die temperature': {'min': 270, 'max': 370},
            'Solidification Time': {'min': 30, 'max': 60},
            'Tilting Angle': {'min': 20, 'max': 35},
            'Tilting Speed': {'min': 15, 'max': 25}
        }
        
        # Get latest data
        df["StrokeStartTime"] = pd.to_datetime(df["StrokeStartTime"], errors='coerce')
        latest_time = df["StrokeStartTime"].max()
        recent_data = df[df["StrokeStartTime"] >= latest_time - timedelta(hours=1)]
        
        for _, row in recent_data.iterrows():
            characteristic = row["ProcessCharacteristics"]
            value = row["AvgValue"]
            machine = row["MachineName"]
            
            # Check if characteristic has defined thresholds
            for thresh_name, thresh_values in thresholds.items():
                if thresh_name.lower() in characteristic.lower():
                    if pd.notna(value):
                        if value < thresh_values['min'] or value > thresh_values['max']:
                            severity = "high" if value < thresh_values['min'] * 0.9 or value > thresh_values['max'] * 1.1 else "medium"
                            
                            alerts.append({
                                "id": alert_id,
                                "machine": machine,
                                "parameter": characteristic,
                                "value": float(value),
                                "threshold": f"{thresh_values['min']}-{thresh_values['max']}",
                                "severity": severity,
                                "time": row["StrokeStartTime"].strftime("%H:%M") if pd.notna(row["StrokeStartTime"]) else "N/A"
                            })
                            alert_id += 1
        
        return jsonify({
            "success": True,
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Alerts error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Auto-train model on startup if Excel file exists
    if os.path.exists("T-13.xlsx"):
        print("Training model on startup...")
        success, message = predictor.train_model()
        print(f"Training result: {message}")
    else:
        print("T-13.xlsx not found. Model will need to be trained manually.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)