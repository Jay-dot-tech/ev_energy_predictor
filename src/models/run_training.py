"""
Run complete model training pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing import EVDataPreprocessor
from train_models import EVModelTrainer

def main():
    """
    Main function to run the complete training pipeline
    """
    print("=== EV ENERGY CONSUMPTION PREDICTION - MODEL TRAINING ===")
    
    # Initialize preprocessor
    preprocessor = EVDataPreprocessor()
    
    # Load and preprocess data
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ev_energy_data.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(data_path)
    
    if X_train is None:
        print("Failed to preprocess data. Exiting.")
        return
    
    # Initialize model trainer
    trainer = EVModelTrainer()
    
    # Train all models
    models, evaluation_results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Create comparison plots
    comparison_fig = trainer.plot_model_comparison()
    if comparison_fig:
        plots_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        comparison_fig.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {plots_dir}/model_comparison.png")
    
    # Plot feature importance for best model
    if trainer.best_model and hasattr(trainer.best_model, 'feature_importances_'):
        feature_importance_fig = trainer.plot_feature_importance(
            trainer.best_model, 
            trainer.best_model_name, 
            X_train.columns.tolist()
        )
        if feature_importance_fig:
            feature_importance_fig.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {plots_dir}/feature_importance.png")
    
    # Save models
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    trainer.save_models(models_dir)
    
    # Print detailed results
    print("\n" + "="*80)
    print("DETAILED MODEL EVALUATION RESULTS")
    print("="*80)
    
    results_df = pd.DataFrame(evaluation_results).T
    print(results_df.round(4))
    
    # Save results to CSV
    results_path = os.path.join(models_dir, 'model_results.csv')
    results_df.to_csv(results_path)
    print(f"\nResults saved to: {results_path}")
    
    # Print best model summary
    print(f"\n" + "="*80)
    print("BEST MODEL SUMMARY")
    print("="*80)
    print(f"Best Model: {trainer.best_model_name}")
    print(f"Best RMSE: {evaluation_results[trainer.best_model_name]['rmse']:.4f}")
    print(f"Best MAE: {evaluation_results[trainer.best_model_name]['mae']:.4f}")
    print(f"Best R²: {evaluation_results[trainer.best_model_name]['r2']:.4f}")
    print(f"Best MAPE: {evaluation_results[trainer.best_model_name]['mape']:.2f}%")
    
    # Create prediction examples
    print(f"\n" + "="*80)
    print("PREDICTION EXAMPLES")
    print("="*80)
    
    # Get a few test samples
    test_samples = X_test.head(5)
    actual_values = y_test.head(5)
    predicted_values = trainer.best_model.predict(test_samples)
    
    print("Sample Predictions vs Actual Values:")
    for i, (actual, predicted) in enumerate(zip(actual_values, predicted_values)):
        print(f"Sample {i+1}: Actual={actual:.3f} kWh, Predicted={predicted:.3f} kWh, Error={abs(actual-predicted):.3f} kWh")
    
    print(f"\nTraining pipeline completed successfully!")
    print(f"Models saved in: {models_dir}")
    print(f"Plots saved in: {plots_dir}")

if __name__ == "__main__":
    main()
