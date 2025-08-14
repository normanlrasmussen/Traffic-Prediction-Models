# Traffic Prediction Models - Machine Learning Project

## 🚗 Project Overview

This project demonstrates a comprehensive analysis of traffic prediction using multiple machine learning approaches. I implemented and compared 6 different models to predict traffic conditions (Heavy, High, Normal, Low) based on vehicle counts, time, and day-of-week features.

## 📊 Results Summary

The project achieved **exceptional performance** across all models, with the best performing model achieving:

- **Accuracy: 99.66%** (XGBoost)
- **Balanced Accuracy: 99.85%** (XGBoost)
- **Precision: 99.67%** (XGBoost)
- **Recall: 99.66%** (XGBoost)
- **F1-Score: 99.66%** (XGBoost)

### Model Performance Comparison

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score |
|-------|----------|-------------------|-----------|--------|----------|
| **XGBoost** | **99.66%** | **99.85%** | **99.67%** | **99.66%** | **99.66%** |
| Gradient Boosting | 99.37% | 99.35% | 99.37% | 99.37% | 99.37% |
| Random Forest | 97.44% | 95.26% | 97.47% | 97.44% | 97.36% |
| Neural Network (PyTorch) | 93.32% | 91.03% | 93.37% | 93.32% | 93.22% |
| Support Vector Classifier | 89.50% | 86.67% | 89.66% | 89.50% | 89.42% |
| Logistic Regression | 89.25% | 84.64% | 89.13% | 89.25% | 89.11% |

## 🛠️ Technical Implementation

### Models Implemented

1. **XGBoost Classifier** - Best performing model with optimized hyperparameters
2. **Gradient Boosting Classifier** - Strong ensemble method with fine-tuned parameters
3. **Random Forest Classifier** - Robust ensemble method with feature importance analysis
4. **Neural Network (PyTorch)** - Custom 3-layer architecture with GELU activation and dropout
5. **Support Vector Classifier** - RBF kernel with optimized C and gamma parameters
6. **Logistic Regression** - Baseline model with L2 regularization

### Key Technical Features

- **Feature Engineering**: Cyclical encoding for time and date features using sine/cosine transformations
- **Data Preprocessing**: Comprehensive pipeline with StandardScaler and OneHotEncoder
- **Hyperparameter Optimization**: Extensive parameter tuning across all models
- **Cross-Validation**: Proper train/test splits with consistent random states
- **Performance Metrics**: Multi-metric evaluation (Accuracy, Balanced Accuracy, Precision, Recall, F1)

### Neural Network Architecture

```python
class TrafficNN(nn.Module):
    def __init__(self):
        super(TrafficNN, self).__init__()
        self.layer1 = nn.Linear(10, 200)    # Input to hidden layer
        self.layer2 = nn.Linear(200, 30)    # Hidden layer
        self.layer3 = nn.Linear(30, 4)      # Output layer (4 traffic classes)
        self.activation = nn.GELU()         # Modern activation function
        self.dropout = nn.Dropout(p=0.1)    # Regularization
```

## 📁 Project Structure

```
Traffic-Prediction-Models/
├── AllModelComp.ipynb              # Main comparison and results
├── data/
│   ├── Traffic.csv                 # Main dataset
│   ├── TrafficTwoMonth.csv         # Extended dataset
│   └── DataInfo.txt               # Dataset documentation
├── Pipeline.*.ipynb                # Individual model implementations
├── PyTorch.*.ipynb                 # Neural network experiments
└── dfMaker.ipynb                   # Data preprocessing utilities
```

## 🔬 Detailed Analysis Notebooks

### Pipeline Models
- `Pipeline.Base.ipynb` - Base pipeline implementation
- `Pipeline.XGB.ipynb` - XGBoost optimization
- `Pipeline.GradientBoosted.ipynb` - Gradient Boosting analysis
- `Pipeline.RandomForest.ipynb` - Random Forest implementation
- `Pipeline.SVC.ipynb` - Support Vector Classifier
- `Pipeline.LogReg.ipynb` - Logistic Regression baseline

### Neural Network Experiments
- `PyTorch.BASE.ipynb` - Base neural network implementation
- `PyTorchNN.NodesComp.ipynb` - Node count optimization
- `PyTorchNN.LRcomp.ipynb` - Learning rate comparison
- `PyTorchNN.EPOHcomp.ipynb` - Epoch optimization
- `PyTorchNN.DropoutComp.ipynb` - Dropout rate analysis
- `PyTorchNN.Decaycomp.ipynb` - Learning rate decay experiments
- `PyTorchNN.OPTIMComp.ipynb` - Optimizer comparison

## 📈 Dataset Information

The project uses a comprehensive traffic dataset containing:
- **Vehicle Counts**: Cars, Bikes, Buses, Trucks
- **Temporal Features**: Date, Time, Day of Week
- **Target Variable**: Traffic Situation (Heavy/High/Normal/Low)
- **Data Collection**: Computer vision-based detection every 15 minutes
- **Time Span**: One month of continuous monitoring

## 🎯 Key Achievements

1. **Exceptional Model Performance**: Achieved 99.66% accuracy with XGBoost
2. **Comprehensive Model Comparison**: Implemented and compared 6 different ML approaches
3. **Advanced Feature Engineering**: Cyclical encoding for temporal features
4. **Production-Ready Pipelines**: Robust preprocessing and model pipelines
5. **Deep Learning Implementation**: Custom PyTorch neural network architecture
6. **Hyperparameter Optimization**: Extensive tuning across all models

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Scikit-learn**: Traditional ML models and pipelines
- **PyTorch**: Deep learning implementation
- **XGBoost**: Gradient boosting framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Jupyter Notebooks**: Interactive development and analysis

## 🚀 Getting Started

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn torch xgboost matplotlib
   ```
3. Open `AllModelComp.ipynb` to see the main results and comparisons
4. Explore individual model notebooks for detailed implementations

## 📊 Visualization

The project includes comprehensive visualizations showing:
- Model performance comparisons across all metrics
- Feature importance analysis
- Training curves and convergence plots
- Hyperparameter optimization results

---

*This project demonstrates strong proficiency in machine learning, data science, and software engineering principles, showcasing the ability to implement, optimize, and compare multiple ML approaches on real-world data.*
