"""
Machine Learning Models Training Module for EV Energy Consumption Prediction
Implements and trains multiple ML models with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any

class EVModelTrainer:
    """
    Handles training and evaluation of ML models for EV energy consumption prediction
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.evaluation_results = {}
        
    def train_linear_regression(self, X_train, y_train):
        """
        Train Linear Regression model
        
        Parameters:
        - X_train: Training features
        - y_train: Training target
        
        Returns:
        - Trained model
        """
        print("\n=== TRAINING LINEAR REGRESSION ===")
        
        # Create and train model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        print(f"Linear Regression trained successfully!")
        print(f"Cross-validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
        
        return lr_model
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest Regressor with hyperparameter tuning
        
        Parameters:
        - X_train: Training features
        - y_train: Training target
        
        Returns:
        - Trained model
        """
        print("\n=== TRAINING RANDOM FOREST ===")
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create Random Forest model
        rf_model = RandomForestRegressor(random_state=42)
        
        # Perform Grid Search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_rf = grid_search.best_estimator_
        
        print(f"Random Forest trained successfully!")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
        
        return best_rf
    
    def train_xgboost(self, X_train, y_train):
        """
        Train XGBoost Regressor with hyperparameter tuning
        
        Parameters:
        - X_train: Training features
        - y_train: Training target
        
        Returns:
        - Trained model
        """
        print("\n=== TRAINING XGBOOST ===")
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Create XGBoost model
        xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
        
        # Perform Grid Search with cross-validation
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_xgb = grid_search.best_estimator_
        
        print(f"XGBoost trained successfully!")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
        
        return best_xgb
    
    def train_gradient_boosting(self, X_train, y_train):
        """
        Train Gradient Boosting Regressor with hyperparameter tuning
        
        Parameters:
        - X_train: Training features
        - y_train: Training target
        
        Returns:
        - Trained model
        """
        print("\n=== TRAINING GRADIENT BOOSTING ===")
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Create Gradient Boosting model
        gb_model = GradientBoostingRegressor(random_state=42)
        
        # Perform Grid Search with cross-validation
        grid_search = GridSearchCV(
            estimator=gb_model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_gb = grid_search.best_estimator_
        
        print(f"Gradient Boosting trained successfully!")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
        
        return best_gb
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate model performance
        
        Parameters:
        - model: Trained model
        - X_test: Test features
        - y_test: Test target
        - model_name: Name of the model
        
        Returns:
        - Dictionary with evaluation metrics
        """
        print(f"\n=== EVALUATING {model_name.upper()} ===")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        print(f"Model: {model_name}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return results
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models and evaluate them
        
        Parameters:
        - X_train: Training features
        - y_train: Training target
        - X_test: Test features
        - y_test: Test target
        
        Returns:
        - Dictionary with all trained models and their evaluation results
        """
        print("\n=== TRAINING ALL MODELS ===")
        
        # Train models
        self.models['linear_regression'] = self.train_linear_regression(X_train, y_train)
        self.models['random_forest'] = self.train_random_forest(X_train, y_train)
        self.models['xgboost'] = self.train_xgboost(X_train, y_train)
        self.models['gradient_boosting'] = self.train_gradient_boosting(X_train, y_train)
        
        # Evaluate all models
        for model_name, model in self.models.items():
            results = self.evaluate_model(model, X_test, y_test, model_name)
            self.evaluation_results[model_name] = results
        
        # Find best model based on RMSE
        best_model_name = min(self.evaluation_results.keys(), 
                            key=lambda x: self.evaluation_results[x]['rmse'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\n=== BEST MODEL ===")
        print(f"Best performing model: {best_model_name}")
        print(f"Best RMSE: {self.evaluation_results[best_model_name]['rmse']:.4f}")
        
        return self.models, self.evaluation_results
    
    def plot_model_comparison(self):
        """
        Plot comparison of model performance
        
        Returns:
        - matplotlib figure
        """
        if not self.evaluation_results:
            print("No evaluation results available. Train models first.")
            return None
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.evaluation_results).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # RMSE comparison
        axes[0, 0].bar(comparison_df.index, comparison_df['rmse'])
        axes[0, 0].set_title('RMSE Comparison (Lower is Better)')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[0, 1].bar(comparison_df.index, comparison_df['mae'])
        axes[0, 1].set_title('MAE Comparison (Lower is Better)')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1, 0].bar(comparison_df.index, comparison_df['r2'])
        axes[1, 0].set_title('R² Score Comparison (Higher is Better)')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        axes[1, 1].bar(comparison_df.index, comparison_df['mape'])
        axes[1, 1].set_title('MAPE Comparison (Lower is Better)')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, model, model_name, feature_names):
        """
        Plot feature importance for tree-based models
        
        Parameters:
        - model: Trained model
        - model_name: Name of the model
        - feature_names: List of feature names
        
        Returns:
        - matplotlib figure
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"{model_name} does not have feature importance.")
            return None
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Create dataframe for plotting
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature')
        plt.title(f'Feature Importance - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        
        return plt.gcf()
    
    def save_models(self, save_dir):
        """
        Save all trained models
        
        Parameters:
        - save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f'{model_name}.pkl')
            joblib.dump(model, model_path)
            print(f"Model saved: {model_path}")
        
        # Save best model separately
        if self.best_model:
            best_model_path = os.path.join(save_dir, 'best_model.pkl')
            joblib.dump(self.best_model, best_model_path)
            print(f"Best model saved: {best_model_path}")
        
        # Save evaluation results
        results_path = os.path.join(save_dir, 'evaluation_results.pkl')
        joblib.dump(self.evaluation_results, results_path)
        print(f"Evaluation results saved: {results_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model
        
        Parameters:
        - model_path: Path to the saved model
        
        Returns:
        - Loaded model
        """
        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return model

if __name__ == "__main__":
    # Example usage
    from src.data.preprocessing import EVDataPreprocessor
    
    # Load preprocessed data
    preprocessor = EVDataPreprocessor()
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    preprocessor.load_preprocessor(os.path.join(models_dir, 'preprocessor.pkl'))
    
    # For demonstration, we'll create some dummy data
    # In practice, you would load the actual preprocessed data
    print("Note: This is a demonstration. In practice, load the actual preprocessed data.")
    
    # Create model trainer
    trainer = EVModelTrainer()
    
    print("Model trainer initialized. Ready to train models with preprocessed data.")
