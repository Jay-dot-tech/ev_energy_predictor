// EV Energy Consumption Predictor - Frontend JavaScript

class EnergyPredictor {
    constructor() {
        this.apiBaseUrl = '/api';
        this.modelInfo = null;
        this.predictionHistory = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadModelInfo();
        this.loadPredictionHistory();
    }

    setupEventListeners() {
        // Form submission
        document.getElementById('predictionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.predictEnergyConsumption();
        });

        // Input validation
        const inputs = document.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('input', () => this.validateInput(input));
            input.addEventListener('change', () => this.validateInput(input));
        });

        // Real-time updates for insights
        const realTimeInputs = ['distance', 'speed', 'temperature'];
        realTimeInputs.forEach(inputId => {
            document.getElementById(inputId).addEventListener('input', () => {
                this.updateEnergyInsights();
            });
        });
    }

    validateInput(input) {
        const value = parseFloat(input.value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        
        if (isNaN(value) || value < min || value > max) {
            input.classList.add('is-invalid');
            this.showInputError(input, `Value must be between ${min} and ${max}`);
        } else {
            input.classList.remove('is-invalid');
            this.hideInputError(input);
        }
    }

    showInputError(input, message) {
        let errorDiv = input.parentNode.querySelector('.invalid-feedback');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.className = 'invalid-feedback';
            input.parentNode.appendChild(errorDiv);
        }
        errorDiv.textContent = message;
    }

    hideInputError(input) {
        const errorDiv = input.parentNode.querySelector('.invalid-feedback');
        if (errorDiv) {
            errorDiv.remove();
        }
    }

    async predictEnergyConsumption() {
        const formData = this.getFormData();
        
        if (!this.validateForm()) {
            this.showAlert('Please correct the errors in the form', 'danger');
            return;
        }

        this.showLoading(true);
        this.hideResults();

        try {
            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Prediction failed');
            }

            const result = await response.json();
            this.displayResults(result);
            this.addToPredictionHistory(result);
            this.showAlert('Prediction completed successfully!', 'success');

        } catch (error) {
            console.error('Prediction error:', error);
            this.showAlert(`Error: ${error.message}`, 'danger');
        } finally {
            this.showLoading(false);
        }
    }

    getFormData() {
        return {
            distance_km: parseFloat(document.getElementById('distance').value),
            avg_speed_kmh: parseFloat(document.getElementById('speed').value),
            road_type: document.getElementById('roadType').value,
            vehicle_load_kg: parseFloat(document.getElementById('load').value),
            outside_temp_celsius: parseFloat(document.getElementById('temperature').value),
            driving_style: document.getElementById('drivingStyle').value
        };
    }

    validateForm() {
        const inputs = document.querySelectorAll('#predictionForm input, #predictionForm select');
        let isValid = true;

        inputs.forEach(input => {
            if (input.hasAttribute('required') && !input.value) {
                input.classList.add('is-invalid');
                isValid = false;
            } else {
                input.classList.remove('is-invalid');
            }
        });

        return isValid;
    }

    displayResults(result) {
        // Update predicted value
        document.getElementById('predictedValue').textContent = result.predicted_energy_consumption_kwh.toFixed(3);
        
        // Update confidence level
        const confidenceElement = document.getElementById('confidenceLevel');
        confidenceElement.textContent = result.confidence.charAt(0).toUpperCase() + result.confidence.slice(1);
        confidenceElement.className = `fw-bold confidence-${result.confidence}`;
        
        // Update energy insights
        this.updateEnergyInsights(result);
        
        // Show results with animation
        this.showResults();
    }

    updateEnergyInsights(result = null) {
        const insightsElement = document.getElementById('energyInsights');
        
        if (result) {
            const distance = parseFloat(document.getElementById('distance').value);
            const speed = parseFloat(document.getElementById('speed').value);
            const consumption = result.predicted_energy_consumption_kwh;
            const efficiency = (distance / consumption).toFixed(2);
            
            let insights = `
                <div class="row">
                    <div class="col-md-6">
                        <p class="mb-1"><strong>Efficiency:</strong> ${efficiency} km/kWh</p>
                        <p class="mb-1"><strong>Energy per km:</strong> ${(consumption / distance).toFixed(3)} kWh/km</p>
                    </div>
                    <div class="col-md-6">
                        <p class="mb-1"><strong>Estimated range:</strong> ${(efficiency * 60).toFixed(0)} km (60kWh battery)</p>
                        <p class="mb-1"><strong>Cost estimate:</strong> $${(consumption * 0.12).toFixed(2)} (@$0.12/kWh)</p>
                    </div>
                </div>
            `;
            
            // Add recommendations based on parameters
            let recommendations = [];
            
            if (speed > 80) {
                recommendations.push("Consider reducing speed for better efficiency");
            }
            if (result.driving_style === 'aggressive') {
                recommendations.push("Eco driving style could save 15-20% energy");
            }
            if (parseFloat(document.getElementById('load').value) > 500) {
                recommendations.push("Reducing load could improve efficiency");
            }
            
            if (recommendations.length > 0) {
                insights += `<div class="mt-2"><strong>Recommendations:</strong><ul class="mb-0">`;
                recommendations.forEach(rec => {
                    insights += `<li>${rec}</li>`;
                });
                insights += `</ul></div>`;
            }
            
            insightsElement.innerHTML = insights;
        } else {
            // Real-time insights without prediction
            const distance = parseFloat(document.getElementById('distance').value) || 0;
            const speed = parseFloat(document.getElementById('speed').value) || 0;
            
            insightsElement.innerHTML = `
                <p class="mb-0">
                    <strong>Current parameters:</strong><br>
                    Distance: ${distance} km, Speed: ${speed} km/h<br>
                    <small>Submit the form to get detailed energy consumption analysis</small>
                </p>
            `;
        }
    }

    showLoading(show) {
        const spinner = document.getElementById('loadingSpinner');
        const predictBtn = document.getElementById('predictBtn');
        
        if (show) {
            spinner.classList.remove('d-none');
            predictBtn.disabled = true;
            predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i> Predicting...';
        } else {
            spinner.classList.add('d-none');
            predictBtn.disabled = false;
            predictBtn.innerHTML = '<i class="fas fa-bolt me-1"></i> Predict Energy Consumption';
        }
    }

    showResults() {
        document.getElementById('welcomeMessage').classList.add('d-none');
        document.getElementById('resultContainer').classList.remove('d-none');
        document.getElementById('resultContainer').classList.add('fade-in');
    }

    hideResults() {
        document.getElementById('resultContainer').classList.add('d-none');
        document.getElementById('welcomeMessage').classList.remove('d-none');
    }

    async loadModelInfo() {
        // Model info display removed - no longer needed
        console.log('Model info loading skipped - display elements removed');
    }

    displayModelInfo(info) {
        // Model info display removed - no longer needed
        console.log('Model info display skipped - elements removed');
    }

    addToPredictionHistory(result) {
        const prediction = {
            timestamp: new Date().toISOString(),
            input: result.input_data,
            output: result.predicted_energy_consumption_kwh,
            confidence: result.confidence
        };
        
        this.predictionHistory.unshift(prediction);
        
        // Keep only last 10 predictions
        if (this.predictionHistory.length > 10) {
            this.predictionHistory = this.predictionHistory.slice(0, 10);
        }
        
        // Save to localStorage
        localStorage.setItem('predictionHistory', JSON.stringify(this.predictionHistory));
    }

    loadPredictionHistory() {
        const saved = localStorage.getItem('predictionHistory');
        if (saved) {
            this.predictionHistory = JSON.parse(saved);
        }
    }

    savePrediction() {
        if (this.predictionHistory.length > 0) {
            const latestPrediction = this.predictionHistory[0];
            const predictionText = `
EV Energy Consumption Prediction
====================================
Date: ${new Date(latestPrediction.timestamp).toLocaleString()}

Input Parameters:
- Distance: ${latestPrediction.input.distance_km} km
- Average Speed: ${latestPrediction.input.avg_speed_kmh} km/h
- Road Type: ${latestPrediction.input.road_type}
- Vehicle Load: ${latestPrediction.input.vehicle_load_kg} kg
- Temperature: ${latestPrediction.input.outside_temp_celsius}°C
- Driving Style: ${latestPrediction.input.driving_style}

Prediction Result:
- Energy Consumption: ${latestPrediction.output} kWh
- Confidence Level: ${latestPrediction.confidence}

Generated by EV Energy Consumption Predictor
            `.trim();

            // Create and download file
            const blob = new Blob([predictionText], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ev_prediction_${new Date().getTime()}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            this.showAlert('Prediction saved successfully!', 'success');
        }
    }

    showAlert(message, type) {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.alert');
        existingAlerts.forEach(alert => alert.remove());
        
        // Create new alert
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 1050; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    resetForm() {
        document.getElementById('predictionForm').reset();
        this.hideResults();
        
        // Clear validation states
        const inputs = document.querySelectorAll('.is-invalid');
        inputs.forEach(input => input.classList.remove('is-invalid'));
        
        // Clear alerts
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(alert => alert.remove());
    }
}

// Utility functions
function resetForm() {
    window.energyPredictor.resetForm();
}

function savePrediction() {
    window.energyPredictor.savePrediction();
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.energyPredictor = new EnergyPredictor();
    
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const form = document.getElementById('predictionForm');
            if (form) {
                form.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to reset form
        if (e.key === 'Escape') {
            resetForm();
        }
    });
});

// Service Worker registration for offline support (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}
