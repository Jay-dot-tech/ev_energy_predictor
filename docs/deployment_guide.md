# Deployment Guide - EV Energy Consumption Predictor

## Table of Contents
1. [Local Deployment](#local-deployment)
2. [Cloud Deployment](#cloud-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Troubleshooting](#troubleshooting)
6. [Performance Optimization](#performance-optimization)

---

## 1. Local Deployment

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning repository)
- At least 2GB RAM
- 500MB disk space

### Step-by-Step Installation

#### 1.1 Clone the Repository
```bash
git clone <repository-url>
cd ev_energy_predictor
```

#### 1.2 Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 1.3 Install Dependencies
```bash
pip install -r requirements.txt
```

#### 1.4 Train Models (First Time Only)
```bash
python src/models/run_training.py
```
This will:
- Generate the dataset
- Preprocess the data
- Train all models
- Save the best model and preprocessor

#### 1.5 Run the Application
```bash
python src/api/app.py
```

#### 1.6 Access the Application
Open your web browser and navigate to:
- `http://localhost:5000`

### Verification
To verify the installation:
1. Check if the web interface loads correctly
2. Test a prediction with sample data
3. Verify API endpoints are working:
   ```bash
   curl http://localhost:5000/api/health
   ```

---

## 2. Cloud Deployment

### 2.1 Render Deployment

#### Prerequisites
- Render account (free tier available)
- GitHub repository with your code

#### Steps:
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Create Render Service**
   - Go to [render.com](https://render.com)
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Configure build settings:
     ```
     Build Command: pip install -r requirements.txt
     Start Command: python src/api/app.py
     ```

3. **Environment Variables**
   Add these environment variables in Render dashboard:
   ```
   FLASK_ENV=production
   PORT=5000
   ```

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete
   - Access your app at the provided URL

### 2.2 Railway Deployment

#### Prerequisites
- Railway account
- GitHub repository

#### Steps:
1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Initialize Railway Project**
   ```bash
   railway init
   ```

3. **Configure Railway**
   Create `railway.toml`:
   ```toml
   [build]
   builder = "nixpacks"
   
   [deploy]
   startCommand = "python src/api/app.py"
   healthcheckPath = "/api/health"
   healthcheckTimeout = 100
   restartPolicyType = "on_failure"
   restartPolicyMaxRetries = 10
   ```

4. **Deploy**
   ```bash
   railway up
   ```

### 2.3 Heroku Deployment

#### Prerequisites
- Heroku account
- Heroku CLI

#### Steps:
1. **Install Heroku CLI**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   heroku login
   ```

2. **Create Heroku App**
   ```bash
   heroku create your-app-name
   ```

3. **Create Procfile**
   Create `Procfile` in root directory:
   ```
   web: python src/api/app.py
   ```

4. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### 2.4 AWS EC2 Deployment

#### Prerequisites
- AWS account
- EC2 instance (t2.micro or larger)

#### Steps:
1. **Launch EC2 Instance**
   - Choose Ubuntu 20.04 LTS
   - Configure security group (allow ports 22, 80, 443, 5000)
   - Download key pair

2. **Connect to Instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Setup Environment**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip git
   git clone <repository-url>
   cd ev_energy_predictor
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Train Models**
   ```bash
   python src/models/run_training.py
   ```

5. **Run as Service**
   Create systemd service:
   ```bash
   sudo nano /etc/systemd/system/ev-predictor.service
   ```
   
   Content:
   ```ini
   [Unit]
   Description=EV Energy Predictor
   After=network.target
   
   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/ev_energy_predictor
   Environment="PATH=/home/ubuntu/ev_energy_predictor/venv/bin"
   ExecStart=/home/ubuntu/ev_energy_predictor/venv/bin/python src/api/app.py
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   Start service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start ev-predictor
   sudo systemctl enable ev-predictor
   ```

---

## 3. Docker Deployment

### 3.1 Create Dockerfile

Create `Dockerfile` in root directory:
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models plots

# Train models (optional - can be done in build)
RUN python src/models/run_training.py

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=src/api/app.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "src/api/app.py"]
```

### 3.2 Create docker-compose.yml

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  ev-predictor:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - PORT=5000
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./plots:/app/plots
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 3.3 Build and Run

#### Build Docker Image
```bash
docker build -t ev-energy-predictor .
```

#### Run with Docker Compose
```bash
docker-compose up -d
```

#### Run with Docker
```bash
docker run -d -p 5000:5000 --name ev-predictor ev-energy-predictor
```

### 3.4 Docker Deployment to Cloud

#### Docker Hub
```bash
# Build and tag
docker build -t yourusername/ev-predictor:latest .

# Push to Docker Hub
docker push yourusername/ev-predictor:latest
```

#### Docker Cloud Services
- **AWS ECS**: Elastic Container Service
- **Google Cloud Run**: Serverless container deployment
- **Azure Container Instances**: Simple container hosting

---

## 4. Environment Configuration

### 4.1 Environment Variables

Create `.env` file in root directory:
```env
# Flask Configuration
FLASK_APP=src/api/app.py
FLASK_ENV=production
SECRET_KEY=your-secret-key-here

# Server Configuration
HOST=0.0.0.0
PORT=5000

# Database (if needed)
DATABASE_URL=sqlite:///ev_predictor.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Model Configuration
MODEL_PATH=models/
PREPROCESSOR_PATH=models/preprocessor.pkl
BEST_MODEL_PATH=models/best_model.pkl

# CORS (if needed)
CORS_ORIGINS=*
```

### 4.2 Production Configuration

Create `config.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    
    # Server settings
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/')
    PREPROCESSOR_PATH = os.environ.get('PREPROCESSOR_PATH', 'models/preprocessor.pkl')
    BEST_MODEL_PATH = os.environ.get('BEST_MODEL_PATH', 'models/best_model.pkl')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/app.log')

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = 'WARNING'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
```

### 4.3 Logging Configuration

Update Flask app to use proper logging:
```python
import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(app):
    if not app.debug:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        file_handler = RotatingFileHandler(
            'logs/app.log', 
            maxBytes=10240000, 
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('EV Energy Predictor startup')
```

---

## 5. Troubleshooting

### 5.1 Common Issues

#### Issue: Model not found
**Error**: `FileNotFoundError: models/best_model.pkl not found`
**Solution**: 
```bash
python src/models/run_training.py
```

#### Issue: Port already in use
**Error**: `Address already in use`
**Solution**:
```bash
# Find process using port 5000
netstat -tulpn | grep :5000
# Kill the process
kill -9 <PID>
# Or use different port
python src/api/app.py --port 5001
```

#### Issue: Module not found
**Error**: `ModuleNotFoundError: No module named 'sklearn'`
**Solution**:
```bash
pip install -r requirements.txt
```

#### Issue: Permission denied
**Error**: `PermissionError: [Errno 13] Permission denied`
**Solution**:
```bash
# On Linux/macOS
sudo chmod +x src/api/app.py
# Or run with proper permissions
python src/api/app.py
```

### 5.2 Performance Issues

#### Issue: Slow predictions
**Solutions**:
- Use smaller model (reduce n_estimators)
- Implement model caching
- Use GPU acceleration for XGBoost
- Optimize feature engineering

#### Issue: High memory usage
**Solutions**:
- Reduce model complexity
- Implement model pruning
- Use memory-efficient data structures
- Clear unused variables

### 5.3 Database Issues (if implemented)

#### Issue: Database connection failed
**Solutions**:
- Check database URL configuration
- Verify database server is running
- Check network connectivity
- Validate credentials

---

## 6. Performance Optimization

### 6.1 Application Optimization

#### Code Optimization
```python
# Use caching for model loading
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model():
    return joblib.load('models/best_model.pkl')

# Optimize preprocessing
def preprocess_input_fast(data):
    # Use vectorized operations
    # Avoid unnecessary copies
    # Pre-allocate arrays
    pass
```

#### Model Optimization
```python
# Use smaller Random Forest
rf_model = RandomForestRegressor(
    n_estimators=50,  # Reduced from 100
    max_depth=10,     # Limit depth
    n_jobs=-1         # Use all cores
)
```

### 6.2 Server Optimization

#### Gunicorn (Production Server)
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.api.app:app
```

#### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 6.3 Caching Strategy

#### Redis Caching
```python
import redis
import json

# Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def predict_with_cache(input_data):
    # Create cache key
    cache_key = f"pred:{hash(str(input_data))}"
    
    # Check cache
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result)
    
    # Make prediction
    result = model.predict(input_data)
    
    # Cache result
    redis_client.setex(cache_key, 3600, json.dumps(result))
    
    return result
```

### 6.4 Monitoring and Metrics

#### Health Monitoring
```python
@app.route('/api/metrics')
def metrics():
    return jsonify({
        'model_loaded': model is not None,
        'predictions_made': prediction_count,
        'uptime': uptime_seconds,
        'memory_usage': psutil.virtual_memory().percent,
        'cpu_usage': psutil.cpu_percent()
    })
```

#### Performance Monitoring
- Use Application Performance Monitoring (APM) tools
- Implement custom metrics collection
- Set up alerts for performance issues
- Monitor error rates and response times

---

## Security Considerations

### 7.1 Basic Security

#### Input Validation
```python
from marshmallow import Schema, fields, validate

class PredictionSchema(Schema):
    distance_km = fields.Float(required=True, validate=validate.Range(min=0, max=1000))
    avg_speed_kmh = fields.Float(required=True, validate=validate.Range(min=0, max=200))
    # ... other fields
```

#### Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ... prediction logic
```

### 7.2 HTTPS Configuration

#### SSL Certificate
```bash
# Use Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

#### Flask HTTPS
```python
from flask import Flask
import ssl

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('server.crt', 'server.key')

app.run(host='0.0.0.0', port=443, ssl_context=context)
```

---

## Scaling Considerations

### 8.1 Horizontal Scaling

#### Load Balancing
- Use multiple app instances
- Implement load balancer (Nginx, HAProxy)
- Use container orchestration (Kubernetes)

#### Database Scaling
- Implement read replicas
- Use connection pooling
- Consider NoSQL for high write loads

### 8.2 Vertical Scaling

#### Resource Allocation
- Increase CPU cores for model training
- Add RAM for larger datasets
- Use SSD for faster I/O operations

---

## Backup and Recovery

### 9.1 Data Backup

#### Model Backup
```bash
# Backup models directory
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Upload to cloud storage
aws s3 cp models_backup_$(date +%Y%m%d).tar.gz s3://your-backup-bucket/
```

#### Automated Backup Script
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "/backup/ev_predictor_$DATE.tar.gz" models/ data/
find /backup -name "ev_predictor_*.tar.gz" -mtime +7 -delete
```

### 9.2 Disaster Recovery

#### Recovery Plan
1. **Infrastructure Recovery**: Recreate servers from backups
2. **Data Recovery**: Restore models and data
3. **Application Recovery**: Restart services
4. **Verification**: Test all endpoints
5. **Monitoring**: Ensure system stability

---

## Conclusion

This deployment guide provides comprehensive instructions for deploying the EV Energy Consumption Predictor in various environments. Choose the deployment method that best fits your requirements:

- **Local Development**: Quick testing and development
- **Cloud Deployment**: Scalable production deployment
- **Docker**: Containerized deployment for consistency
- **Bare Metal**: Full control over infrastructure

Remember to:
- Follow security best practices
- Implement proper monitoring
- Set up backup and recovery procedures
- Optimize for your specific use case
- Keep dependencies updated

For additional support or questions, refer to the project documentation or create an issue in the repository.
