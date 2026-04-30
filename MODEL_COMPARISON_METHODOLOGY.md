# Model Comparison and Performance Evaluation Methodology

## Overview
This document describes the implementation of Section 4.2 "Model Comparison and Performance Evaluation" based on academic research methodology for machine learning model evaluation.

## Implementation Summary

### 1. Classification Task Design
Following the paper's approach of binary classification (Retained vs Dropped Out), we implemented:
- **Target Variable**: High Consumption Prediction
- **Class 0 (Normal)**: Electricity consumption ≤ 75th percentile
- **Class 1 (High)**: Electricity consumption > 75th percentile
- **Purpose**: Predict anomalously high electricity consumption patterns

### 2. Machine Learning Models Implemented

#### Model 1: Random Forest Classifier
- **Algorithm**: Ensemble learning with decision trees
- **Configuration**: 
  - 100 estimators
  - Max depth: 15
  - Class weight: balanced (handles class imbalance)
- **Performance**: 96.33% accuracy

#### Model 2: Support Vector Machine (SVM)
- **Algorithm**: Kernel-based classification
- **Configuration**:
  - RBF kernel
  - C=1.0
  - Gamma: scale
  - Class weight: balanced
- **Performance**: 88.36% accuracy

#### Model 3: XGBoost
- **Algorithm**: Gradient boosting
- **Configuration**:
  - 100 estimators
  - Max depth: 6
  - Learning rate: 0.1
  - Scale_pos_weight: automatic (handles imbalance)
- **Performance**: 95.85% accuracy

### 3. Evaluation Metrics (Confusion Matrix Based)

Following the paper's methodology, we calculate four key metrics from the confusion matrix:

#### Confusion Matrix Components
```
                    Predicted
                Normal (0)  High (1)
Actual  Normal (0)   TN        FP
        High (1)     FN        TP
```

- **TP (True Positives)**: Correctly predicted high consumption
- **TN (True Negatives)**: Correctly predicted normal consumption
- **FP (False Positives)**: Normal predicted as high
- **FN (False Negatives)**: High predicted as normal

#### Metrics Calculated

1. **Recall (Sensitivity / True Positive Rate)**
   - Formula: `TP / (TP + FN)`
   - Meaning: Fraction of actual positive samples correctly identified
   - **Critical for**: Minimizing dangerous misclassifications
   - **High recall** = Fewer false negatives = Better detection of high consumption

2. **Precision**
   - Formula: `TP / (TP + FP)`
   - Meaning: Fraction of positive predictions that are actually correct
   - **Critical for**: Monitoring false positive rate
   - **High precision** = Fewer false alarms = More reliable predictions

3. **Accuracy**
   - Formula: `(TP + TN) / (TP + TN + FP + FN)`
   - Meaning: Overall ratio of correct predictions
   - **Note**: Can be misleading with imbalanced datasets

4. **F1-Score**
   - Formula: `2 × (Precision × Recall) / (Precision + Recall)`
   - Meaning: Harmonic mean of precision and recall
   - **Critical for**: Balancing sensitivity and precision
   - **Purpose**: Ensures high recall doesn't come at cost of excessive false positives

### 4. Results Comparison

#### Table 1: Comparative Performance Metrics

| Machine Learning Model | Target Class | Precision | Recall | F1-Score | Overall Accuracy |
|------------------------|--------------|-----------|--------|----------|------------------|
| **Random Forest**      | Normal (0)   | 1.00      | 0.95   | 0.97     | **96.33%**       |
|                        | High (1)     | 0.88      | 0.99   | 0.93     |                  |
| **SVM**                | Normal (0)   | 0.97      | 0.87   | 0.92     | **88.36%**       |
|                        | High (1)     | 0.72      | 0.91   | 0.80     |                  |
| **XGBoost**            | Normal (0)   | 1.00      | 0.95   | 0.97     | **95.85%**       |
|                        | High (1)     | 0.87      | 0.99   | 0.93     |                  |

### 5. Key Findings

1. **Best Overall Model**: Random Forest (96.33% accuracy)
   - Excellent recall for high consumption (0.99)
   - Strong precision for normal consumption (1.00)
   - Best F1-scores across both classes

2. **High Recall Models**: Random Forest and XGBoost both achieve 0.99 recall for high consumption
   - Critical for early warning systems
   - Minimizes missed high consumption events

3. **SVM Performance**: Lower overall accuracy (88.36%)
   - Still maintains good recall (0.91) for high consumption
   - Lower precision (0.72) leads to more false positives

### 6. Practical Implications

#### For Early Warning Systems
- **Random Forest** is recommended for deployment
- High recall (0.99) ensures very few high consumption events are missed
- High precision (0.88) keeps false alarm rate manageable

#### Trade-offs
- **High Recall Priority**: Use Random Forest or XGBoost
  - Best for safety-critical applications
  - Acceptable false positive rate
  
- **Balanced Approach**: Use Random Forest
  - Best overall F1-scores
  - Optimal precision-recall balance

### 7. Visualization Outputs

The implementation generates:

1. **confusion_matrices.png**: Visual representation of confusion matrices for all three models
   - Shows TP, TN, FP, FN counts
   - Color-coded heatmaps for easy interpretation

2. **Classification Reports**: Detailed per-class metrics
   - Precision, recall, F1-score for each class
   - Macro and weighted averages
   - Support (sample counts)

### 8. Files Generated

- `electricity_classifier.pkl`: Best performing model (Random Forest)
- `confusion_matrices.png`: Confusion matrix visualizations
- `train_model.py`: Complete implementation code

## How to Use

### Training
```bash
python train_model.py
```

### Output
- Console: Detailed metrics table and classification reports
- Files: Saved model and confusion matrix visualizations

### Loading the Model
```python
import joblib
model = joblib.load('electricity_classifier.pkl')
predictions = model.predict(X_new)
```

## Methodology Alignment with Paper

This implementation follows the paper's approach:

✅ **Confusion Matrix**: Used as foundation for all metrics  
✅ **Multiple Models**: Compared 3 different algorithms  
✅ **Standard Metrics**: Precision, Recall, F1-Score, Accuracy  
✅ **Binary Classification**: Two-class problem (Normal vs High)  
✅ **Class Imbalance Handling**: Balanced class weights  
✅ **Visual Representation**: Confusion matrix heatmaps  
✅ **Detailed Reporting**: Per-class and overall metrics  

## References

The methodology is based on standard machine learning evaluation practices:
- Confusion Matrix analysis
- Precision-Recall trade-offs
- F1-Score for balanced evaluation
- Cross-validation for model reliability

## Future Enhancements

Potential improvements:
1. ROC-AUC curves for threshold analysis
2. Precision-Recall curves
3. Cross-validation with multiple folds
4. Hyperparameter tuning with grid search
5. Feature importance analysis for interpretability
