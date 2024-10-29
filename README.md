# Wine Quality Classification with Feature Selection

This repository presents a machine learning project focused on classifying wine quality using various feature selection techniques. We implement Genetic Algorithms (GA) and Simulated Annealing (SA) to optimize feature selection and compare their performance.

## 1. Dataset Overview

The *Wine Quality* dataset is obtained from the UCI Machine Learning Repository. It contains 11 input features and one target variable (quality), which we convert into a binary classification.

- **Features**: 11 characteristics related to wine (e.g., acidity, alcohol).
- **Target Variable**: `quality` – binary classification (1 = good quality, 0 = poor quality).
- **Size**: 1599 samples.

### Loading the Dataset

```python
import pandas as pd

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, delimiter=';')

# Binarize target variable for classification
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)

# Dataset overview
df.info()
df.head()
```

## 2. Wrapper Technique Implementation

### Genetic Algorithms (GA)

Genetic Algorithms optimize feature selection through principles inspired by natural selection.

#### Implementation

```python
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import random
import matplotlib.pyplot as plt

# Prepare data
X = df.drop(columns='quality')
y = df['quality']

# GA setup and execution...
```

![Genetic Algorithm Performance](genetic_algorithm_performance.png)

#### Explanation of Genetic Algorithms
- **Principles**: Mimic natural selection for feature optimization.
- **Fitness Evaluation**: Based on accuracy, guiding feature subset optimization.

### Simulated Annealing (SA)

Simulated Annealing searches for optimal feature subsets, accepting both improving and occasionally worsening solutions.

#### Implementation

```python
from sklearn.metrics import accuracy_score
import math

# Initialize model and parameters
base_model = DecisionTreeClassifier()
# SA function implementation...
```

![Simulated Annealing Performance](simulated_annealing_performance.png)

#### Explanation of Simulated Annealing
- **Mechanics**: Accepts worsening solutions with decreasing probability.
- **Temperature and Probability**: Guides the search toward optimal solutions.

## 3. Comparison with Part 1

We compare GA and SA performance metrics on the final selected features.

### Evaluation Metrics
- **GA Metrics**: 
  - Accuracy: 1.0
  - Precision: 1.0
  - Recall: 1.0
  - F1 Score: 1.0

- **SA Metrics**: 
  - Accuracy: 1.0
  - Precision: 1.0
  - Recall: 1.0
  - F1 Score: 1.0

## Visualizations

### Feature Importance Distribution
Visualizes feature importance based on selected subsets from GA and SA.

```python
# Feature importance visualization implementation...
def plot_feature_importance(X, selected_features, title):
    importances = np.mean([cross_val_score(DecisionTreeClassifier(), X.iloc[:, selected_features], y, cv=5, scoring='accuracy') for _ in range(10)])
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(selected_features)), importances, color='blue')
    plt.xticks(range(len(selected_features)), selected_features, rotation=45)
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance (Mean Accuracy)')
    plt.savefig("Feature_Importance_GA_Selected_Features.png")
    plt.savefig("Feature_Importance_SA_Selected_Features.png")
    plt.show()

# Plot feature importance for GA and SA
plot_feature_importance(X, selected_features_ga, "Feature Importance from GA Selected Features")
plot_feature_importance(X, current_state, "Feature Importance from SA Selected Features")
```

![Feature Importance from GA Selected Features](Feature_Importance_GA_Selected_Features.png)
![Feature Importance from SA Selected Features](Feature_Importance_SA_Selected_Features.png)

### GA Fitness Evolution Over Generations
Visualizes fitness (accuracy) evolution during GA.

![GA Fitness Evolution](ga_fitness_evolution.png)

### Temperature vs. Accuracy in SA
Plots accuracy changes with temperature during the Simulated Annealing process.

![SA Temperature vs. Accuracy](sa_temperature_accuracy.png)

### Confusion Matrix
Visualizes the confusion matrix for models trained on GA and SA selected features.

```python
# Confusion matrix implementation...
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(selected_features, title):
    model = DecisionTreeClassifier()
    model.fit(X.iloc[:, selected_features], y)
    y_pred = model.predict(X.iloc[:, selected_features])
    
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig("Confusion_Matrix_GA_Selected_Features.png")
    plt.savefig("Confusion_Matrix_SA_Selected_Features.png")
    plt.show()

# Confusion matrix for GA and SA selected features
plot_confusion_matrix(selected_features_ga, "Confusion Matrix for GA Selected Features")
plot_confusion_matrix(current_state, "Confusion Matrix for SA Selected Features")
```

![Confusion Matrix for GA Selected Features](Confusion_Matrix_GA_Selected_Features.png)
![Confusion Matrix for SA Selected Features](Confusion_Matrix_SA_Selected_Features.png)

### Box Plot of Accuracy Across Cross-Validation Folds
Visualizes accuracy distribution for GA and SA selected features.

```python
# Box plot implementation...
# Box plot for accuracy distribution
def plot_accuracy_distribution(selected_features, title):
    model = DecisionTreeClassifier()
    cv_scores = cross_val_score(model, X.iloc[:, selected_features], y, cv=5)
    
    plt.figure(figsize=(8, 6))
    plt.boxplot(cv_scores, vert=False)
    plt.title(title)
    plt.xlabel("Accuracy")
    plt.savefig("Accuracy_Distribution_GA_Selected_Features.png")
    plt.savefig("Accuracy_Distribution_SA_Selected_Features.png")
    plt.show()

# Accuracy distribution for GA and SA
plot_accuracy_distribution(selected_features_ga, "Accuracy Distribution for GA Selected Features")
plot_accuracy_distribution(current_state, "Accuracy Distribution for SA Selected Features")
```

![Accuracy Distribution for GA Selected Features](Accuracy_Distribution_GA_Selected_Features.png)
![Accuracy Distribution for SA Selected Features](Accuracy_Distribution_SA_Selected_Features.png)

## 4. Conclusion

### Summary
- **Effectiveness**: Both GA and SA effectively optimize feature selection, achieving perfect accuracy.
- **Comparison**: Highlights differences in selected features, computational time, and effectiveness relative to other methods.

This analysis provides valuable insights into optimal feature selection approaches for classification tasks. Feel free to explore the code and adapt it for your needs!
