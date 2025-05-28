
# 🛡️ SMS Spam Detection System

![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-blueviolet )
![Python](https://img.shields.io/badge/Python-3.8%2B-blue )
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange )
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange )
![Dataset](https://img.shields.io/badge/Dataset-SMS%20Spam%20Collection-yellowgreen )

This repository contains a machine learning model for SMS spam detection using the **SMS Spam Collection dataset** with 5,000 messages. The system employs SVM and Naive Bayes algorithms to classify messages as spam or ham with 98% accuracy.

## ✨ Key Features

- **Text preprocessing pipeline**: Lowercasing, punctuation removal, and stopword removal
- **TF-IDF vectorization**: Converts text into numerical features
- **Hyperparameter tuning**: GridSearchCV for optimal model parameters
- **Dual algorithm approach**: Naive Bayes and SVM implementations
- **Custom prediction**: Test your own messages with the model
- **Performance visualization**: Comparative metrics visualization
- **Model comparison**: Detailed evaluation metrics for both algorithms

## 📊 Model Performance

| Model        | F1-Score | Accuracy | Precision | Recall |
|--------------|----------|----------|-----------|--------|
| Naive Bayes  | 0.93     | 0.98     | 0.99      | 0.88   |
| SVM          | 0.91     | 0.98     | 0.94      | 0.89   |

## ℹ️ Dataset Information
- **5,000 SMS messages** (4825 ham, 747 spam)
- Labels: `ham` (legitimate) or `spam`
- Original dataset source: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset )
- Dataset characteristics after cleaning:
  ```
  Label distribution after cleaning:
  0.0    4825
  1.0     747
  ```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install pandas numpy scikit-learn matplotlib nltk
  ```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kaaado/Spam-Detection.git 
   cd Spam-Detection
   ```

### Usage
1. Run the Jupyter notebook:
   ```bash
   jupyter notebook spamdetect.ipynb
   ```
2. The notebook will:
   - Load and preprocess the dataset
   - Perform hyperparameter tuning
   - Train and evaluate both models
   - Generate performance visualizations
   - Allow custom message testing

3. To test with custom messages:
   ```python
   # In the notebook's final cell
   Enter a message to check ➡️: "WIN a FREE vacation! Claim your prize now!"
   🚨 Spam!
   ```

## 💡 Implementation Highlights

### Text Preprocessing
```python
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = ' '.join([word for word in text.split() 
                   if word not in stopwords.words('english')])
    return text
```

### Hyperparameter Tuning
```python
# Naive Bayes tuning
nb_params = {'alpha': [0.01, 0.1, 0.5, 1, 2]}
nb_grid = GridSearchCV(MultinomialNB(), nb_params, cv=5, scoring='f1')

# SVM tuning
svm_params = {'C': [0.1, 0.5, 1, 5, 10]}
svm_grid = GridSearchCV(SVC(kernel='linear'), svm_params, cv=5, scoring='f1')
```

### Model Evaluation
```python
def train_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        print(f"{name} Report:
{classification_report(y_test, y_pred)}")
    return results
```

## 📈 Results Visualization
The notebook generates a comprehensive performance comparison:

![Performance Comparison](https://github.com/kaaado/Spam-Detection/raw/main/performance_comparison.png )

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- SMS Spam Collection Dataset creators
- Scikit-learn and NLTK development teams
- Inspired by real-world spam detection applications

---
**Protect your inbox with AI!** ⚡
