"""
EV Energy Consumption Dataset Generator
Generates synthetic data for training ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def generate_ev_dataset(n_samples=5000, random_state=42):
    """
    Generate synthetic EV energy consumption dataset
    
    Parameters:
    - n_samples: Number of samples to generate
    - random_state: Random seed for reproducibility
    
    Returns:
    - DataFrame with EV energy consumption data
    """
    np.random.seed(random_state)
    
    print(f"Generating {n_samples} samples of EV energy consumption data...")
    
    # Generate features based on real-world EV characteristics
    
    # Distance (km): 1-500 km, with most trips being shorter
    distance = np.random.exponential(scale=30, size=n_samples)
    distance = np.clip(distance, 1, 500)
    
    # Average Speed (km/h): 10-120 km/h
    # City driving tends to have lower speeds
    avg_speed = np.random.normal(50, 20, size=n_samples)
    avg_speed = np.clip(avg_speed, 10, 120)
    
    # Road Type: 0=City, 1=Highway
    # Higher speeds correlate with highway driving
    road_type_prob = 1 / (1 + np.exp(-(avg_speed - 40) / 10))
    road_type = np.random.binomial(1, road_type_prob, size=n_samples)
    
    # Vehicle Load (kg): 0-1000 kg (passengers + cargo)
    vehicle_load = np.random.gamma(2, 100, size=n_samples)
    vehicle_load = np.clip(vehicle_load, 0, 1000)
    
    # Outside Temperature (°C): -20 to 45°C
    # Normal distribution around 20°C
    outside_temp = np.random.normal(20, 10, size=n_samples)
    outside_temp = np.clip(outside_temp, -20, 45)
    
    # Driving Style: 0=Eco, 1=Normal, 2=Aggressive
    # Weighted distribution: 30% Eco, 50% Normal, 20% Aggressive
    driving_style = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.5, 0.2])
    
    # Calculate energy consumption based on physics and real EV characteristics
    # Base consumption: ~0.15 kWh/km for modern EVs
    
    # Base consumption per km
    base_consumption_per_km = 0.15
    
    # Adjust for speed (optimal around 50-60 km/h)
    speed_factor = 1 + 0.5 * np.abs(avg_speed - 55) / 55
    
    # Adjust for road type (highway is more efficient)
    road_factor = np.where(road_type == 1, 0.9, 1.1)
    
    # Adjust for load (more load = more consumption)
    load_factor = 1 + (vehicle_load / 1000) * 0.3
    
    # Adjust for temperature (extreme temps reduce efficiency)
    temp_factor = 1 + 0.2 * np.abs(outside_temp - 20) / 20
    
    # Adjust for driving style
    style_factor = np.array([0.8, 1.0, 1.3])[driving_style]
    
    # Calculate total consumption
    consumption_per_km = base_consumption_per_km * speed_factor * road_factor * load_factor * temp_factor * style_factor
    
    # Add some random noise
    noise = np.random.normal(0, 0.02, size=n_samples)
    consumption_per_km = consumption_per_km + noise
    
    # Total energy consumption (kWh)
    energy_consumption = distance * consumption_per_km
    
    # Ensure positive values
    energy_consumption = np.maximum(energy_consumption, 0.1)
    
    # Create DataFrame
    data = pd.DataFrame({
        'distance_km': distance.round(2),
        'avg_speed_kmh': avg_speed.round(1),
        'road_type': road_type,
        'vehicle_load_kg': vehicle_load.round(0),
        'outside_temp_celsius': outside_temp.round(1),
        'driving_style': driving_style,
        'energy_consumption_kwh': energy_consumption.round(3)
    })
    
    # Convert categorical to meaningful labels
    data['road_type'] = data['road_type'].map({0: 'city', 1: 'highway'})
    data['driving_style'] = data['driving_style'].map({0: 'eco', 1: 'normal', 2: 'aggressive'})
    
    print(f"Dataset generated successfully!")
    print(f"Shape: {data.shape}")
    print(f"\nDataset Statistics:")
    print(data.describe())
    
    return data

def save_dataset(data, filename='ev_energy_data.csv'):
    """Save dataset to CSV file"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    data.to_csv(filepath, index=False)
    print(f"Dataset saved to: {filepath}")
    return filepath

if __name__ == "__main__":
    # Generate dataset
    ev_data = generate_ev_dataset(n_samples=5000, random_state=42)
    
    # Save dataset
    save_dataset(ev_data, 'ev_energy_data.csv')
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(ev_data.head())
    
    # Display data types
    print("\nData types:")
    print(ev_data.dtypes)
    
    # Display value counts for categorical variables
    print("\nRoad Type Distribution:")
    print(ev_data['road_type'].value_counts())
    
    print("\nDriving Style Distribution:")
    print(ev_data['driving_style'].value_counts())
