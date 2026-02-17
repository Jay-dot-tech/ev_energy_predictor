# EV Energy Consumption Predictor

A comprehensive Machine Learning project for predicting Electric Vehicle (EV) energy consumption using advanced ML algorithms.

## 🚗 Project Overview

This project implements a complete end-to-end ML solution to predict the energy consumption (in kWh) of electric vehicles based on various driving and environmental parameters. The system uses multiple ML models and provides a user-friendly web interface for real-time predictions.

## ✨ Features

- **Multiple ML Models**: Linear Regression, Random Forest, XGBoost, Gradient Boosting
- **Best Model Selection**: Automatically selects the best performing model (Random Forest with R² = 0.9995)
- **Interactive Web Interface**: Clean, responsive UI with real-time predictions
- **REST API**: Flask-based backend for programmatic access
- **Comprehensive Evaluation**: RMSE, MAE, R², MAPE metrics
- **Feature Engineering**: Advanced feature creation and preprocessing
- **Data Visualization**: Model comparison and feature importance plots

## 🎯 Problem Statement

Predict the energy consumption (kWh) of an Electric Vehicle based on:
- Distance (km)
- Average Speed (km/h)
- Road Type (city / highway)
- Vehicle Load (kg)
- Outside Temperature (°C)
- Driving Style (eco / normal / aggressive)

## 📊 Dataset

The project uses a synthetic dataset of 5,000 samples with realistic EV energy consumption patterns. The data includes:
- **Numerical Features**: Distance, Speed, Load, Temperature
- **Categorical Features**: Road Type, Driving Style
- **Target Variable**: Energy Consumption (kWh)

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API     │    │   ML Models     │
│                 │    │                 │    │                 │
│ • HTML/CSS/JS   │◄──►│ • Prediction    │◄──►│ • Random Forest │
│ • Bootstrap     │    │ • Validation    │    │ • XGBoost      │
│ • Interactive   │    │ • Model Info    │    │ • Linear Reg.   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Preprocessing │
                       │                 │
                       │ • Scaling       │
                       │ • Encoding      │
                       │ • Feature Eng.  │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ev_energy_predictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the models**
   ```bash
   python src/models/run_training.py
   ```

5. **Run the application**
   ```bash
   python src/api/app.py
   ```

6. **Open the application**
   Navigate to `http://localhost:5000` in your browser

## 📁 Project Structure

```
ev_energy_predictor/
├── src/
│   ├── data/
│   │   ├── generate_dataset.py      # Dataset generation
│   │   └── preprocessing.py          # Data preprocessing
│   ├── models/
│   │   ├── train_models.py           # ML model training
│   │   └── run_training.py           # Training pipeline
│   ├── api/
│   │   └── app.py                    # Flask API server
│   └── utils/
├── frontend/
│   ├── templates/
│   │   └── index.html               # Main UI template
│   └── static/
│       ├── css/style.css            # Custom styles
│       └── js/main.js               # Frontend JavaScript
├── data/
│   └── ev_energy_data.csv           # Generated dataset
├── models/
│   ├── best_model.pkl               # Best trained model
│   ├── preprocessor.pkl             # Data preprocessor
│   └── model_results.csv            # Evaluation results
├── plots/
│   ├── model_comparison.png         # Model performance comparison
│   └── feature_importance.png       # Feature importance plot
├── docs/
├── tests/
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

## 🤖 Machine Learning Models

### Models Implemented

1. **Linear Regression**
   - Baseline model for comparison
   - RMSE: 1.9516
   - R²: 0.8964

2. **Random Forest Regressor** ⭐ *Best Model*
   - Hyperparameter tuned with GridSearchCV
   - RMSE: 0.1391
   - R²: 0.9995
   - MAE: 0.0726

3. **XGBoost Regressor**
   - Advanced gradient boosting
   - RMSE: 0.5171
   - R²: 0.9927

4. **Gradient Boosting Regressor**
   - Ensemble method
   - RMSE: 0.1628
   - R²: 0.9993

### Feature Engineering

- **Interaction Features**: Speed × Load, Distance × Temperature
- **Polynomial Features**: Speed², Distance²
- **Efficiency Score**: Distance / Energy Consumption
- **Categorical Encoding**: Label encoding for road type and driving style

### Evaluation Metrics

- **RMSE** (Root Mean Square Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **R²** (R-squared): Higher is better (0 to 1)
- **MAPE** (Mean Absolute Percentage Error): Lower is better

## 🔌 API Endpoints

### Prediction API

**POST** `/api/predict`

```json
{
  "distance_km": 50.0,
  "avg_speed_kmh": 60.0,
  "road_type": "highway",
  "vehicle_load_kg": 200.0,
  "outside_temp_celsius": 25.0,
  "driving_style": "normal"
}
```

**Response:**
```json
{
  "predicted_energy_consumption_kwh": 7.5,
  "confidence": "high",
  "input_data": {...},
  "timestamp": "2024-01-01T12:00:00",
  "model_info": {
    "model_type": "Random Forest",
    "features_used": 11
  }
}
```

### Other Endpoints

- **GET** `/api/health` - Health check
- **GET** `/api/model_info` - Model information and performance metrics
- **POST** `/api/batch_predict` - Batch prediction for multiple inputs

## 🎨 Frontend Features

### Interactive Form
- Real-time input validation
- Parameter range checking
- User-friendly interface with icons and descriptions

### Results Display
- Predicted energy consumption in kWh
- Confidence level indicator
- Energy efficiency insights
- Cost estimation
- Range estimation for standard EV battery

### Additional Features
- Save predictions to file
- Model performance display
- Responsive design for mobile devices
- Real-time energy insights

## 📈 Model Performance

The Random Forest model achieved exceptional performance:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| RMSE | 0.1391 kWh | Very low prediction error |
| MAE | 0.0726 kWh | Excellent accuracy |
| R² | 0.9995 | Near-perfect fit |
| MAPE | 1.63% | Very low percentage error |

## 🛠️ Development

### Adding New Features

1. **New Input Parameters**: Update the preprocessing pipeline and model training
2. **New Models**: Add to `train_models.py` and update the training pipeline
3. **New Visualizations**: Add to the frontend JavaScript and CSS

### Testing

```bash
# Run unit tests (when implemented)
python -m pytest tests/

# Test API endpoints
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 50, "avg_speed_kmh": 60, "road_type": "highway", "vehicle_load_kg": 200, "outside_temp_celsius": 25, "driving_style": "normal"}'
```

## 🚀 Deployment

### Local Deployment

1. Install dependencies
2. Train models
3. Run Flask server
4. Access via `http://localhost:5000`

### Cloud Deployment (Optional)

The application is ready for deployment on:
- **Render**: Simple web app deployment
- **Railway**: Container-based deployment
- **Heroku**: Platform as a Service
- **AWS EC2**: Virtual machine deployment

### Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "src/api/app.py"]
```

## 📊 Results and Insights

### Key Findings

1. **Random Forest** performs best with near-perfect R² score
2. **Distance** and **Speed** are the most important features
3. **Driving Style** significantly impacts energy consumption
4. **Temperature** affects battery efficiency
5. **Road Type** (city vs highway) influences consumption patterns

### Feature Importance

1. Distance (primary factor)
2. Speed × Load interaction
3. Average Speed
4. Temperature
5. Driving Style

## 🎓 Academic Use

This project is designed as a complete final-year AIML project with:

- **Problem Statement**: Clear, real-world application
- **Methodology**: Complete ML pipeline
- **Results**: Comprehensive evaluation
- **Documentation**: Academic-ready reports
- **Viva Preparation**: Questions and answers included

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or support, please contact:
- **Project Developer**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [your-username]

## 🙏 Acknowledgments

- Scikit-learn for ML algorithms
- Flask for web framework
- Bootstrap for UI components
- XGBoost for gradient boosting
- All open-source contributors

---

**Note**: This project uses synthetic data for demonstration purposes. For production use, replace with real EV energy consumption data.
