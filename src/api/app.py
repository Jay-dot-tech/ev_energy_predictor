"""
Flask API for EV Energy Consumption Prediction
Provides REST endpoints for energy consumption prediction
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing import EVDataPreprocessor

app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessor
model = None
preprocessor = None

def load_model_and_preprocessor():
    """
    Load the trained model and preprocessor
    """
    global model, preprocessor
    
    try:
        # Get the base directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # Load preprocessor
        preprocessor_path = os.path.join(base_dir, 'models', 'preprocessor.pkl')
        preprocessor = EVDataPreprocessor()
        preprocessor.load_preprocessor(preprocessor_path)
        
        # Load best model
        model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
        model = joblib.load(model_path)
        
        print("Model and preprocessor loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model/preprocessor: {e}")
        return False

def validate_input(data):
    """
    Validate input data
    
    Parameters:
    - data: Input dictionary
    
    Returns:
    - Tuple of (is_valid, error_message)
    """
    required_fields = [
        'distance_km', 'avg_speed_kmh', 'road_type', 
        'vehicle_load_kg', 'outside_temp_celsius', 'driving_style'
    ]
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate data types and ranges
    try:
        distance = float(data['distance_km'])
        if distance <= 0 or distance > 1000:
            return False, "Distance must be between 0 and 1000 km"
        
        speed = float(data['avg_speed_kmh'])
        if speed <= 0 or speed > 200:
            return False, "Average speed must be between 0 and 200 km/h"
        
        load = float(data['vehicle_load_kg'])
        if load < 0 or load > 2000:
            return False, "Vehicle load must be between 0 and 2000 kg"
        
        temp = float(data['outside_temp_celsius'])
        if temp < -50 or temp > 60:
            return False, "Temperature must be between -50 and 60°C"
        
        road_type = data['road_type'].lower()
        if road_type not in ['city', 'highway']:
            return False, "Road type must be 'city' or 'highway'"
        
        driving_style = data['driving_style'].lower()
        if driving_style not in ['eco', 'normal', 'aggressive']:
            return False, "Driving style must be 'eco', 'normal', or 'aggressive'"
        
    except (ValueError, TypeError):
        return False, "Invalid data type in input fields"
    
    return True, ""

def preprocess_input(data):
    """
    Preprocess input data for prediction
    
    Parameters:
    - data: Input dictionary
    
    Returns:
    - Preprocessed features DataFrame
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([data])
    
    # Apply the same preprocessing steps as training data
    # Encode categorical features
    input_df['road_type'] = preprocessor.label_encoders['road_type'].transform(input_df['road_type'])
    input_df['driving_style'] = preprocessor.label_encoders['driving_style'].transform(input_df['driving_style'])
    
    # Feature engineering
    input_df['speed_load_interaction'] = input_df['avg_speed_kmh'] * input_df['vehicle_load_kg']
    input_df['distance_temp_interaction'] = input_df['distance_km'] * input_df['outside_temp_celsius']
    input_df['speed_squared'] = input_df['avg_speed_kmh'] ** 2
    input_df['distance_squared'] = input_df['distance_km'] ** 2
    
    # Note: efficiency_score cannot be calculated without energy_consumption_kwh
    # We'll set it to 0 for prediction (it won't be used by the model)
    input_df['efficiency_score'] = 0
    
    # Ensure all expected columns are present
    expected_columns = preprocessor.feature_columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Select and order columns correctly
    input_df = input_df[expected_columns]
    
    # Scale features
    input_scaled = preprocessor.scaler.transform(input_df)
    
    return pd.DataFrame(input_scaled, columns=expected_columns)

@app.route('/')
def index():
    """
    Serve the main page
    """
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict energy consumption endpoint
    
    Expected input:
    {
        "distance_km": 50.0,
        "avg_speed_kmh": 60.0,
        "road_type": "highway",
        "vehicle_load_kg": 200.0,
        "outside_temp_celsius": 25.0,
        "driving_style": "normal"
    }
    
    Output:
    {
        "predicted_energy_consumption_kwh": 7.5,
        "confidence": "high",
        "input_data": {...},
        "timestamp": "2024-01-01T12:00:00"
    }
    """
    try:
        # Check if model is loaded
        if model is None or preprocessor is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please try again later'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No input data provided'
            }), 400
        
        # Validate input
        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({
                'error': 'Invalid input',
                'message': error_message
            }), 400
        
        # Preprocess input
        processed_input = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        
        # Ensure prediction is positive
        prediction = max(0.1, prediction)
        
        # Determine confidence based on input ranges
        confidence = "high"
        distance = float(data['distance_km'])
        speed = float(data['avg_speed_kmh'])
        
        # Lower confidence for extreme values
        if distance > 200 or speed > 120 or speed < 20:
            confidence = "medium"
        if distance > 400 or speed > 150 or speed < 10:
            confidence = "low"
        
        # Prepare response
        response = {
            'predicted_energy_consumption_kwh': round(prediction, 3),
            'confidence': confidence,
            'input_data': data,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_type': 'Random Forest',
                'features_used': len(preprocessor.feature_columns)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model
    """
    if model is None or preprocessor is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    try:
        # Load evaluation results
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        results_path = os.path.join(base_dir, 'models', 'evaluation_results.pkl')
        evaluation_results = joblib.load(results_path)
        
        # Get best model info
        best_model_name = None
        best_rmse = float('inf')
        
        for model_name, results in evaluation_results.items():
            if results['rmse'] < best_rmse:
                best_rmse = results['rmse']
                best_model_name = model_name
        
        return jsonify({
            'model_type': best_model_name.replace('_', ' ').title(),
            'performance_metrics': evaluation_results.get(best_model_name, {}),
            'features': preprocessor.feature_columns,
            'categorical_encodings': {
                'road_type': dict(zip(preprocessor.label_encoders['road_type'].classes_, 
                                    preprocessor.label_encoders['road_type'].transform(preprocessor.label_encoders['road_type'].classes_))),
                'driving_style': dict(zip(preprocessor.label_encoders['driving_style'].classes_, 
                                        preprocessor.label_encoders['driving_style'].transform(preprocessor.label_encoders['driving_style'].classes_)))
            },
            'training_data_info': {
                'total_samples': 5000,
                'features_count': len(preprocessor.feature_columns),
                'target_variable': preprocessor.target_column
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to get model info',
            'message': str(e)
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple inputs
    
    Expected input:
    {
        "predictions": [
            {
                "distance_km": 50.0,
                "avg_speed_kmh": 60.0,
                ...
            },
            ...
        ]
    }
    """
    try:
        if model is None or preprocessor is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
        
        data = request.get_json()
        
        if not data or 'predictions' not in data:
            return jsonify({
                'error': 'No prediction data provided'
            }), 400
        
        predictions_data = data['predictions']
        
        if not isinstance(predictions_data, list):
            return jsonify({
                'error': 'Predictions must be an array'
            }), 400
        
        results = []
        
        for i, pred_data in enumerate(predictions_data):
            # Validate each input
            is_valid, error_message = validate_input(pred_data)
            if not is_valid:
                results.append({
                    'index': i,
                    'error': 'Invalid input',
                    'message': error_message
                })
                continue
            
            # Preprocess and predict
            try:
                processed_input = preprocess_input(pred_data)
                prediction = model.predict(processed_input)[0]
                prediction = max(0.1, prediction)
                
                results.append({
                    'index': i,
                    'predicted_energy_consumption_kwh': round(prediction, 3),
                    'input_data': pred_data
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'error': 'Prediction failed',
                    'message': str(e)
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(predictions_data),
            'successful_predictions': len([r for r in results if 'predicted_energy_consumption_kwh' in r]),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """
    Main function to initialize and run the Flask app
    """
    # Load model and preprocessor
    if not load_model_and_preprocessor():
        print("Failed to load model and preprocessor. Exiting.")
        return
    
    # Configure template and static folders
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    app.template_folder = os.path.join(frontend_dir, 'frontend', 'templates')
    app.static_folder = os.path.join(frontend_dir, 'frontend', 'static')
    
    # Add template folder to Jinja search path
    app.jinja_loader.searchpath.append(app.template_folder)
    
    # Run the Flask app
    print("Starting Flask API server...")
    print(f"Frontend templates: {app.template_folder}")
    print(f"Frontend static: {app.static_folder}")
    print(f"Template files: {os.listdir(app.template_folder)}")
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
