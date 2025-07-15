# California Housing Price Prediction & Fundamentals

## Project Overview

This comprehensive project demonstrates a full-cycle machine learning initiative aimed at predicting median house values in California districts. It highlights practical skills in data manipulation, model building, and evaluation using industry-standard libraries. Critically, it is complemented by a "Linear Regression from Scratch" implementation, showcasing a deep foundational understanding of core ML algorithms beyond just library usage.

This dual approach provides a strong case for establishing an in-house AI/ML team, illustrating both the application of robust frameworks and the capability to understand, debug, and innovate at the algorithmic level, essential for tackling diverse and complex business challenges.

## Features

This project encompasses the following key stages and functionalities:

### 1. Data Acquisition & Initial Exploration

* **Automated Data Loading:** Implements a robust function (`load_housing_data`) to download the California Housing dataset from a specified URL (`https://github.com/ageron/data/raw/main/housing.tgz`) and extract it locally if not already present. This ensures reproducible data access.
* **Initial Data Inspection:** Utilizes `pandas` methods (`.head()`, `.info()`, `.describe()`, `.value_counts()`) for a preliminary understanding of the dataset's structure, data types, summary statistics, and identifying potential issues like missing values or cardinality of categorical features.
    * **Identified Challenges:** Specifically addresses the presence of missing values in the `total_bedrooms` column and the categorical nature of `ocean_proximity`.

### 2. Feature Engineering

* **Deriving New Features:** Creates three new, potentially more informative features from existing ones to help models capture more complex relationships:
    * `rooms_per_household`: Total rooms divided by households.
    * `bedrooms_per_room`: Total bedrooms divided by total rooms.
    * `population_per_household`: Population divided by households.
    These features provide ratios that can be more predictive than raw counts.

### 3. Data Preprocessing Pipelines

* **Handling Missing Values:** Employs `sklearn.impute.SimpleImputer` with a `strategy='median'` to fill missing values in numerical columns, particularly `total_bedrooms`, ensuring that the model can process complete data.
* **Categorical Encoding:** Transforms the `ocean_proximity` categorical feature into a numerical format using `sklearn.preprocessing.OneHotEncoder`, which creates binary (dummy) variables, making it compatible with machine learning algorithms.
* **Numerical Scaling:** Applies `sklearn.preprocessing.StandardScaler` to numerical features. This crucial step scales features to a common range (zero mean and unit variance), preventing features with larger values from dominating the learning process and ensuring optimal performance for gradient-based models.
* **Automated Data Transformation with `ColumnTransformer` & `Pipeline`:**
    * Leverages `sklearn.compose.ColumnTransformer` to apply different preprocessing steps to different subsets of columns (numerical vs. categorical).
    * Constructs a `sklearn.pipeline.Pipeline` that sequences the imputation, scaling, and encoding steps. This ensures all transformations are applied consistently and correctly, simplifying the workflow and preventing data leakage between training and testing sets.

### 4. Model Training & Evaluation

* **Diverse Model Experimentation:** A wide range of regression models are trained and compared to identify the most suitable algorithm for the task:
    * `LinearRegression`
    * `DecisionTreeRegressor`
    * `RandomForestRegressor`
    * `GradientBoostingRegressor`
    * `HistGradientBoostingRegressor`
* **Robust Evaluation with Cross-Validation:**
    * The primary evaluation metric is **Root Mean Squared Error (RMSE)**, which provides a measure of the average magnitude of the errors in predictions, expressed in the same units as the target variable (median house value).
    * Model performance is rigorously assessed using `sklearn.model_selection.cross_val_score`. This technique divides the data into multiple folds, trains the model on a subset of the folds, and validates on the remaining fold, repeating the process. This provides a more reliable estimate of the model's generalization performance than a single train-test split.
* **Hyperparameter Tuning with `GridSearchCV`:**
    * The `RandomForestRegressor`, often a strong performer, is further optimized using `sklearn.model_selection.GridSearchCV`. This method systematically searches over a predefined grid of hyperparameter values (e.g., `n_estimators`, `max_features`, `max_leaf_nodes`) to find the combination that yields the best cross-validation performance. This rigorous tuning significantly improves model accuracy and robustness.

### 5. Model Persistence & Confidence Intervals

* **Model Saving:** The final, optimized model (`final_model`) is saved to disk using `joblib.dump` (`my_california_housing_model.pkl`). This allows for easy loading and deployment of the trained model without needing to retrain it, essential for productionizing ML solutions.
* **Confidence Interval Calculation:** Demonstrates the calculation of confidence intervals for the squared errors of predictions using `scipy.stats.t.interval`. This provides a statistical understanding of the uncertainty associated with the model's predictions, crucial for conveying reliability to stakeholders.

## Complementary Project: Linear Regression from Scratch

Beyond the application of existing libraries, this project is complemented by a separate implementation of **Linear Regression from scratch**. This includes:

* **Manual Implementation:** Developing the core components of Linear Regression (e.g., cost function, gradient descent algorithm, prediction function) using fundamental numerical libraries like NumPy, without relying on `scikit-learn`'s pre-built models.
* **Demonstrates Foundational Understanding:** This effort showcases a deep comprehension of the underlying mathematical principles and algorithmic mechanics of machine learning. It underscores the ability to:
    * Understand how models learn and make predictions at a fundamental level.
    * Debug and troubleshoot issues more effectively by understanding internal workings.
    * Potentially customize or adapt algorithms for unique business needs.
    * Build a strong theoretical basis for future advanced ML endeavors.

This combination of practical application with foundational knowledge provides a robust skillset for any emerging AI/ML team.

## Project Structure
```bash
    ├── datasets/                 # Directory to store the downloaded housing data
│   └── housing/
│       └── housing.csv       # The raw California housing dataset
│   └── housing.tgz           # Compressed housing dataset tarball
├── regression.ipynb          # Jupyter Notebook with all project code
└── my_california_housing_model.pkl # Saved final trained model
```

## Installation and Setup

To run this project locally, ensure you have Python installed (preferably Python 3.8+). Then, install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib scipy jupyter