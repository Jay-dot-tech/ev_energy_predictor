"""
Data Preprocessing Module for EV Energy Consumption Prediction
Handles data cleaning, feature engineering, and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

class EVDataPreprocessor:
    """
    Handles preprocessing of EV energy consumption data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = None
        self.target_column = 'energy_consumption_kwh'
        
    def load_data(self, filepath):
        """
        Load dataset from CSV file
        
        Parameters:
        - filepath: Path to the CSV file
        
        Returns:
        - DataFrame with loaded data
        """
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self, data):
        """
        Perform basic data exploration
        
        Parameters:
        - data: DataFrame to explore
        """
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset Shape: {data.shape}")
        print(f"\nColumn Names: {list(data.columns)}")
        
        print(f"\nData Types:")
        print(data.dtypes)
        
        print(f"\nMissing Values:")
        print(data.isnull().sum())
        
        print(f"\nBasic Statistics:")
        print(data.describe())
        
        print(f"\nCategorical Variables:")
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"\n{col} value counts:")
            print(data[col].value_counts())
    
    def clean_data(self, data):
        """
        Clean the dataset
        
        Parameters:
        - data: DataFrame to clean
        
        Returns:
        - Cleaned DataFrame
        """
        print("\n=== DATA CLEANING ===")
        
        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Check for missing values
        missing_values = cleaned_data.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Found missing values:\n{missing_values}")
            
            # Handle missing values for numerical columns
            numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            cleaned_data[numerical_cols] = self.imputer.fit_transform(cleaned_data[numerical_cols])
            
            # Handle missing values for categorical columns
            categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if cleaned_data[col].isnull().sum() > 0:
                    mode_value = cleaned_data[col].mode()[0]
                    cleaned_data[col].fillna(mode_value, inplace=True)
            
            print("Missing values handled.")
        else:
            print("No missing values found.")
        
        # Check for outliers in numerical columns
        numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != self.target_column:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = cleaned_data[(cleaned_data[col] < lower_bound) | 
                                       (cleaned_data[col] > upper_bound)]
                
                if len(outliers) > 0:
                    print(f"Found {len(outliers)} outliers in {col}")
                    # Cap outliers instead of removing them
                    cleaned_data[col] = np.clip(cleaned_data[col], lower_bound, upper_bound)
        
        print("Data cleaning completed.")
        return cleaned_data
    
    def encode_categorical_features(self, data):
        """
        Encode categorical features
        
        Parameters:
        - data: DataFrame with categorical features
        
        Returns:
        - DataFrame with encoded categorical features
        """
        print("\n=== CATEGORICAL ENCODING ===")
        
        encoded_data = data.copy()
        categorical_cols = encoded_data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                encoded_data[col] = self.label_encoders[col].fit_transform(encoded_data[col])
                print(f"Encoded {col}: {dict(zip(self.label_encoders[col].classes_, 
                                                self.label_encoders[col].transform(self.label_encoders[col].classes_)))}")
            else:
                encoded_data[col] = self.label_encoders[col].transform(encoded_data[col])
        
        print("Categorical encoding completed.")
        return encoded_data
    
    def feature_engineering(self, data):
        """
        Perform feature engineering
        
        Parameters:
        - data: DataFrame for feature engineering
        
        Returns:
        - DataFrame with engineered features
        """
        print("\n=== FEATURE ENGINEERING ===")
        
        engineered_data = data.copy()
        
        # Create interaction features
        engineered_data['speed_load_interaction'] = engineered_data['avg_speed_kmh'] * engineered_data['vehicle_load_kg']
        engineered_data['distance_temp_interaction'] = engineered_data['distance_km'] * engineered_data['outside_temp_celsius']
        
        # Create polynomial features for important variables
        engineered_data['speed_squared'] = engineered_data['avg_speed_kmh'] ** 2
        engineered_data['distance_squared'] = engineered_data['distance_km'] ** 2
        
        # Create efficiency features
        engineered_data['efficiency_score'] = engineered_data['distance_km'] / (engineered_data['energy_consumption_kwh'] + 0.001)
        
        print("Feature engineering completed.")
        print(f"New features added: speed_load_interaction, distance_temp_interaction, speed_squared, distance_squared, efficiency_score")
        
        return engineered_data
    
    def prepare_features_and_target(self, data):
        """
        Separate features and target
        
        Parameters:
        - data: DataFrame with features and target
        
        Returns:
        - X: Features DataFrame
        - y: Target Series
        """
        # Separate features and target
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        print(f"Features prepared: {len(self.feature_columns)} features")
        print(f"Target variable: {self.target_column}")
        
        return X, y
    
    def scale_features(self, X, fit=True):
        """
        Scale numerical features
        
        Parameters:
        - X: Features DataFrame
        - fit: Whether to fit the scaler (True for training, False for prediction)
        
        Returns:
        - Scaled features DataFrame
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            print("Feature scaler fitted and applied.")
        else:
            X_scaled = self.scaler.transform(X)
            print("Feature scaler applied.")
        
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Parameters:
        - X: Features DataFrame
        - y: Target Series
        - test_size: Proportion of data for testing
        - random_state: Random seed
        
        Returns:
        - X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Data split completed:")
        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, filepath, test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline
        
        Parameters:
        - filepath: Path to the raw data file
        - test_size: Proportion of data for testing
        - random_state: Random seed
        
        Returns:
        - X_train, X_test, y_train, y_test
        """
        print("=== STARTING PREPROCESSING PIPELINE ===")
        
        # Load data
        data = self.load_data(filepath)
        if data is None:
            return None, None, None, None
        
        # Explore data
        self.explore_data(data)
        
        # Clean data
        cleaned_data = self.clean_data(data)
        
        # Encode categorical features
        encoded_data = self.encode_categorical_features(cleaned_data)
        
        # Feature engineering
        engineered_data = self.feature_engineering(encoded_data)
        
        # Prepare features and target
        X, y = self.prepare_features_and_target(engineered_data)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)
        
        # Scale features
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_test_scaled = self.scale_features(X_test, fit=False)
        
        print("=== PREPROCESSING PIPELINE COMPLETED ===")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, filepath):
        """
        Save the preprocessor object
        
        Parameters:
        - filepath: Path to save the preprocessor
        """
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to: {filepath}")
    
    def load_preprocessor(self, filepath):
        """
        Load the preprocessor object
        
        Parameters:
        - filepath: Path to the saved preprocessor
        """
        preprocessor_data = joblib.load(filepath)
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.imputer = preprocessor_data['imputer']
        self.feature_columns = preprocessor_data['feature_columns']
        self.target_column = preprocessor_data['target_column']
        print(f"Preprocessor loaded from: {filepath}")

if __name__ == "__main__":
    # Example usage
    preprocessor = EVDataPreprocessor()
    
    # Path to the generated dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ev_energy_data.csv')
    
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(data_path)
    
    # Save preprocessor for later use
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    preprocessor.save_preprocessor(os.path.join(models_dir, 'preprocessor.pkl'))
    
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
