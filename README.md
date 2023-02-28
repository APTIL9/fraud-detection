# Fraud Detection using Support Vector Machine
This project aims to build a fraud detection system using Support Vector Machine (SVM) to identify fraudulent transactions in a dataset of credit card transactions. The dataset used in this project contains transactions made by European cardholders in September 2013, where 492 out of 284,807 transactions are fraudulent.

## Dataset
The dataset used in this project is the [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset from Kaggle. It contains 31 columns, where the features V1 to V28 are the principal components obtained with PCA, and the remaining columns are 'Time', 'Amount', and 'Class'. The 'Class' column is the target variable, where 1 represents a fraudulent transaction and 0 represents a non-fraudulent transaction.

## Functions
The following functions were implemented in this project:

- load_data(): Load the dataset from a CSV file and return a pandas dataframe.
- explore_data(): Perform exploratory data analysis on the dataset, including checking for missing values, data types, and class distribution, as well as visualizing the data using histograms and boxplots.
- preprocess_data(): Preprocess the dataset by scaling the numerical features using StandardScaler, splitting the data into training and testing sets, and balancing the classes using SMOTE.
- train_svm(): Train a SVM model using GridSearchCV to find the best hyperparameters and return the model and its performance metrics.
- train_ensemble(): Train an ensemble model using Random Forest and Gradient Boosting classifiers, and return the best model and its performance metrics.
- plot_roc_curve(): Plot the ROC curve and calculate the AUC score for a given model.

## Results
Using SVM, the model achieved an accuracy score of 0.831, while the ensemble model achieved an accuracy of 0.947. The performance of the models was evaluated using confusion matrices, classification reports, and ROC curves. The best model was chosen based on its ability to minimize false positives and false negatives.

## Usage
To use the functions in this project, simply clone the repository and run the main.py file. The main.py file calls the functions in the correct order and prints the results to the console. You can also modify the hyperparameters and other settings in the config.py file to experiment with different configurations.

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



