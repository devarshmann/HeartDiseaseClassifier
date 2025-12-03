# Heart Disease Classification Analysis - CMPT 459 Data Mining Project

A comprehensive data mining analysis of the Heart Disease Health Indicators dataset using clustering, outlier detection, feature selection, classification, and hyperparameter tuning techniques.

## Overview

This project analyzes 253,680 health indicator samples to identify factors causing heart disease and develop a predictive model suitable for medical screening applications. The final Random Forest classifier achieves 98% recall and 0.83 AUC-ROC.

## Repository Structure

```
HeartDiseaseClassifier/
├── Analysis_Pipeline.ipynb               # Main analysis notebook (all parts)
├── EDA_Preprocessing.ipynb              # Initial exploratory data analysis
├── REPORT.tex                           # Final 2-page LaTeX report
├── heart_disease_health_indicators_BRFSS2015.csv  # Dataset
├── figures/                             # Generated visualization figures
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── hyperparameter_comparison.png
├── cluster_analysis/                    # Part 1: Clustering module
│   ├── __init__.py
│   └── kmeans_analyzer.py
├── outlier_detection/                   # Part 2: Outlier detection module
│   ├── __init__.py
│   └── isolation_forest_analyzer.py
├── feature_selection/                   # Part 3: Feature selection module
│   ├── __init__.py
│   └── mutual_info_selector.py
├── classification/                      # Part 4: Classification module
│   ├── __init__.py
│   └── random_forest_classifier.py
└── hyperparameter_tuning/               # Part 5: Hyperparameter tuning module
    ├── __init__.py
    └── random_forest_tuner.py
```

## Environment Setup

### Requirements

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd HeartDiseaseClassifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Required Python Packages

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning algorithms
- `scipy` - Scientific computing (for Random Search distributions)
- `jupyter` - Notebook environment

## Data Access

**Important**: The dataset file must be named exactly `heart_disease_health_indicators_BRFSS2015.csv` and placed in the root directory of the project for the Jupyter notebook to run correctly.

**To download and set up the dataset:**

1. Download the dataset from: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
2. Save the downloaded file as `heart_disease_health_indicators_BRFSS2015.csv`
3. Place the file in the root directory of the project (same folder as `Analysis_Pipeline.ipynb`)

## How to Run

### Main Analysis Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Analysis_Pipeline.ipynb` in the browser

3. Run cells sequentially:
   - **Cell 0-2**: Setup and data loading
   - **Part 1**: Cluster Analysis (Cells 3-20)
   - **Part 2**: Outlier Detection (Cells 21-30)
   - **Part 3**: Feature Selection (Cells 31-33)
   - **Part 4**: Classification (Cells 34-46)
   - **Part 5**: Hyperparameter Tuning (Cells 47-55)

4. Each part can be run independently, but they build on previous results

### Running Individual Modules

The project uses a modular structure. Each module can be imported and used independently:

```python
# Example: Using the classification module
from classification import RandomForestClassifierAnalyzer

# Initialize with your data
classifier = RandomForestClassifierAnalyzer(X, y, feature_names=feature_names)

# Split data
classifier.split_data(test_size=0.2)

# Train model
model = classifier.train(n_estimators=100, class_weight='balanced')

# Evaluate
metrics = classifier.evaluate(cv=5)
classifier.print_evaluation()
```

### Generating Figures

Figures are automatically generated when running the notebook cells. To save figures for the report:

1. Run the visualization cells in `Analysis_Pipeline.ipynb`
2. Figures will be displayed in the notebook
3. Right-click on figures and "Save Image As..." to save to `figures/` directory
4. Required figures:
   - `confusion_matrix.png`
   - `roc_curve.png`
   - `hyperparameter_comparison.png`

### Compiling the Report

1. Ensure all figures are in the `figures/` directory
2. Compile the LaTeX report:
```bash
pdflatex REPORT.tex
```

Or use an online LaTeX compiler like Overleaf.

## Project Components

### Part 1: Cluster Analysis
- **Module**: `cluster_analysis/kmeans_analyzer.py`
- **Method**: K-Means clustering
- **Output**: Optimal clusters (k=2), cluster visualizations, cluster-target relationships

### Part 2: Outlier Detection
- **Module**: `outlier_detection/isolation_forest_analyzer.py`
- **Method**: Isolation Forest
- **Output**: Outlier identification, outlier analysis, decision on keeping/removing outliers

### Part 3: Feature Selection
- **Module**: `feature_selection/mutual_info_selector.py`
- **Method**: Mutual Information
- **Output**: Top 10 selected features, feature importance scores, performance comparison

### Part 4: Classification
- **Module**: `classification/random_forest_classifier.py`
- **Method**: Random Forest with class balancing
- **Output**: Trained model, evaluation metrics, confusion matrix, ROC curve

### Part 5: Hyperparameter Tuning
- **Module**: `hyperparameter_tuning/random_forest_tuner.py`
- **Method**: Random Search
- **Output**: Optimized hyperparameters, before/after comparison, performance improvements

## Key Results

- **Final Model Performance**: 98% recall, 0.83 AUC-ROC
- **Top Risk Factors**: General Health Status, Age, High Blood Pressure, High Cholesterol
- **Best Use Case**: Medical screening applications where missing cases is critical

## Notes

- The dataset has severe class imbalance (9.4% positive class), which is handled through class weighting and threshold optimization
- Hyperparameter tuning uses a 30k sample subset for computational efficiency, then retrains on full data
- All modules use `random_state=42` for reproducibility
- The project is designed for solo work (one algorithm per part as per requirements)

## Troubleshooting

**Issue**: Module not found errors
- **Solution**: Ensure you're running from the project root directory and all `__init__.py` files are present

**Issue**: Memory errors during hyperparameter tuning
- **Solution**: Reduce `TUNING_SAMPLE_SIZE` in Cell 48 or use fewer iterations in Random Search

**Issue**: Figures not displaying
- **Solution**: Ensure matplotlib backend is set correctly. Try `%matplotlib inline` in notebook cells

Arshdeep Mann

CMPT 459 Data Mining Project - Fall 2025
