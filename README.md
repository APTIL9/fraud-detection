# Fraud Detection using Support Vector Machine
This project aims to build a fraud detection system using various Machine Learning algorithms to identify fraudulent transactions in a dataset of credit card transactions. The dataset used in this project contains transactions made by European cardholders in September 2013, where 492 out of 284,807 transactions are fraudulent.

## Dataset
The dataset used in this project is the [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset from Kaggle. It contains 31 columns, where the features V1 to V28 are the principal components obtained with PCA, and the remaining columns are 'Time', 'Amount', and 'Class'. The 'Class' column is the target variable, where 1 represents a fraudulent transaction and 0 represents a non-fraudulent transaction.

## Results
Using Random Forest, the model achieved an accuracy score of 0.99, while the Gradient Boosting model achieved an accuracy of 0.98. The performance of the models was evaluated using confusion matrices, classification reports, and ROC curves. The best model was chosen based on its ability to minimize false positives and false negatives.

## Usage
To use the functions in this project, simply clone the repository and run the main.py file. The main.py file calls the functions in the correct order and prints the results to the console. You can also modify the hyperparameters and other settings in the config.py file to experiment with different configurations. And download the dataset from the above given link.

## Dependencies
This project requires the following Python libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- imblearn

These libraries can be installed using pip:

```python
pip install numpy pandas matplotlib seaborn scikit-learn imblearn
```

License
This project is licensed under the MIT License.



