# AI-ML-Crime-Rate-Prediction-Model
# Introduction:

  This project implements a machine learning solution to predict the violent crime rate (ViolentCrimesPerPop) in different communities. It utilizes an ensemble of three robust gradient boosting algorithms: LightGBM, XGBoost, and CatBoost. The model follows a clear and systematic data science pipeline, from data preprocessing to final prediction generation, ensuring high accuracy and reliability.

---

# Features:

* Ensemble Modeling: Combines the predictions from three distinct boosting models (LightGBM, XGBoost, CatBoost) to create a more stable and accurate final result.

* K-Fold Cross-Validation: The model is trained and validated using a 5-fold cross-validation strategy. This technique helps in preventing overfitting and provides a more robust estimate of the model's performance on unseen data.

* Automated Data Preprocessing: Missing values in the dataset are automatically handled and imputed using the median value of each column.

* Efficient Prediction Generation: The script is designed to run and generate a submission.csv file with the final predictions for the test dataset.
  
---

# Modules Used:

* numpy: A fundamental package for scientific computing in Python, used for numerical operations and array handling.

* pandas: An essential library for data manipulation and analysis, used for loading and working with the datasets.

* scikit-learn: A powerful library for machine learning, providing tools for model selection (KFold), data preprocessing (SimpleImputer), and performance metrics (mean_squared_error).

* lightgbm: An efficient and fast gradient boosting framework.

* xgboost: A popular and highly optimized gradient boosting implementation.

* catboost: A robust gradient boosting library that automatically handles categorical features.

* matplotlib and seaborn: Libraries for creating static, animated, and interactive visualizations in Python, often used for data analysis.

* warnings: Used to manage and filter warning messages for cleaner console output.
  
  ---
  
# How It Works:

* Data Loading: The script begins by loading the new_train.csv and new_test_for_participants.csv files into pandas DataFrames.

* Preprocessing: It identifies and handles any missing values in the datasets by replacing them with the median of their respective columns using SimpleImputer.

* Model Training: The core of the project is a cross-validation loop. In each of the 5 folds, the three models are trained on a portion of the data. Early stopping is used during training to prevent the models from running for too long and to find the optimal number of boosting rounds.

* Final Prediction: The final output is a weighted average of the predictions from each model. This ensemble strategy is designed to minimize error and produce a more balanced result. The weights are set as:

  * LightGBM: 50%

  * XGBoost: 30%

  * CatBoost: 20%
    
---

# Logic Behind It:

* The approach of using a weighted average ensemble is based on the principle that combining multiple "weak learners" can create a powerful "strong learner."

* Why an Ensemble?

  A single model, no matter how sophisticated, can have biases and may not capture all the complex patterns within the data. By combining three different boosting algorithms, each with its own unique approach to building trees and handling features, we can mitigate the weaknesses of any single model. This leads to a more robust, stable, and less-biased final prediction, which is crucial for achieving a lower overall error (RMSE).

*  Why These Weights?

    The specific weights (50/30/20) are chosen based on the typical performance characteristics of these models. LightGBM is often the fastest and can achieve a very good baseline performance, so it is given the highest weight. XGBoost and CatBoost provide complementary predictive power, helping to refine the final prediction. These weights are a common starting point in many data science competitions and can be further optimized through hyperparameter tuning.
* Importance of Cross-Validation:
  
   K-Fold cross-validation is a critical step to ensure that the model generalizes well to unseen data. By training and validating the model on different subsets of the data, we get a much more reliable estimate of its true performance. This prevents the model from simply memorizing the training data (overfitting) and failing when faced with new, real-world examples.
  
---

# Dataset Description:

The project utilizes a dataset commonly used for predicting crime rates, containing a rich set of demographic, socioeconomic, and housing-related features for various communities.

  * new_train.csv: The training dataset, including the target variable ViolentCrimesPerPop.

  * new_test_for_participants.csv: The test dataset for which predictions need to be generated.

---

# Key Columns and Their Meanings:

  ID: A unique numerical identifier for each community.

  ViolentCrimesPerPop: The target variable for prediction, representing the violent crime rate per 100,000 population.

  population: The total population of the community.

  householdsize: The average number of people per household.

  agePct12t21: The percentage of the population between the ages of 12 and 21.

  medIncome: The median household income.

  PctPopUnderPov: The percentage of the population living below the poverty line.

  PctUnemployed: The percentage of the population that is unemployed.

  PctBSorMore: The percentage of people with a Bachelor's degree or higher.

  PctImmigRecent: The percentage of recent immigrants in the community.

  (and etc refer the file column..)

---

# Setup Instructions

1. Clone Repository
   
    To get started, clone the project repository from GitHub:

 ```bash git
  clone https://github.com/Reneshb24/AI-ML-Crime-Rate-Prediction-Model.git
  
  cd AI-ML-Crime-Rate-Prediction-Model
```


2. Install Dependencies
   
    Open your terminal or command prompt and install all the required Python libraries using pip:

  ```bash git
  pip install numpy pandas scikit-learn lightgbm xgboost catboost matplotlib seaborn
```


3. File Placement
   
    Ensure that all necessary files (AI-ML-Crime-Rate-Prediction-Model.py, new_train.csv, and new_test_for_participants.csv) are located in the same directory for the script to run correctly.

5. A Note on XGBoost Version Compatibility
   
    Please be aware that some older versions of the XGBoost library may not support the callbacks parameter used for early stopping in Model.py. To ensure smooth execution, it is highly recommended to update your XGBoost library to the latest version by running the following command:

  ```bash git
  pip install --upgrade xgboost
```
---

# Running the Project

  * To run the script and generate the predictions, navigate to the project directory in your terminal and execute:
    
```bash git
    python AI-ML-Crime-Rate-Prediction-Model.py
```
---

# Key Functions and Details:

The Model.py script is structured to perform the entire machine learning pipeline. The key functions and components include:

* Data Loading: Using pd.read_csv().

* Imputation: SimpleImputer(strategy="median") to handle missing values.

* Cross-Validation: KFold(n_splits=5, shuffle=True, random_state=42) for robust model training.

* Model Training: The fit() method is called on each model (model_lgb, model_xgb, model_cat).

* Early Stopping: Used with lightgbm and xgboost to optimize the number of boosting rounds.

* Prediction: The predict() method is used on the trained models to make predictions on the validation and test sets.

* Ensemble Prediction: Predictions are combined using a weighted average.

* Output: The final predictions are saved to submission.csv using df.to_csv().

---

# Output:

After the script has successfully run, it will produce the following output:

* Console Output: The progress and RMSE for each fold will be displayed. The final Overall Out-of-Fold RMSE will be printed, giving a measure of the model's overall performance.

* submission.csv: A CSV file containing the ID and the predicted ViolentCrimesPerPop for each community in the test dataset.

---

# Future Enhancements:

* Feature Engineering: Exploring and creating new features from the existing data to improve model performance.

* Hyperparameter Tuning: Using techniques like Grid Search or Bayesian Optimization to find the optimal parameters for each model.

* Advanced Ensemble: Implementing a stacking or blending approach for a more sophisticated ensemble model.

* Model Deployment: Creating an API or web application to serve the model's predictions.

* Real-time Data: Integrating the model with real-time data sources for continuous prediction updates.

---

# Author:

Renesh B

---
