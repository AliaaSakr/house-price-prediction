# House Price Prediction via Advanced Model Stacking
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
##  Project Overview
This project focuses on building an end-to-end data science workflow to predict residential house prices. The core objective was to handle complex, high-dimensional housing data through rigorous feature engineering and robust ensemble modeling.

##  Key Methodologies & Workflow
* **Exploratory Data Analysis (EDA):** Identified skewness, handled missing values, and treated outliers in housing features.
* **Feature Engineering & Reduction:** Created custom interaction terms and applied Principal Component Analysis (PCA) to reduce dimensionality while preserving variance.
* **Pipeline Construction:** Built reproducible Scikit-Learn pipelines to prevent data leakage during preprocessing.
* **Model Stacking:** Combined multiple base learners using a Meta-Regressor to optimize prediction accuracy ($R^2$ score).
* **Hyperparameter Tuning:** Automated optimization using `GridSearchCV`.

* ## Results & Key Takeaways
* **Quantifiable Performance Boost:** The stacked ensemble architecture achieved a superior $R^2$ score and lower Root Mean Squared Error (RMSE) compared to any single baseline model (such as baseline linear regression).
* **Dimensionality Control:** Successfully used Principal Component Analysis (PCA) to condense highly correlated housing features, capturing over 90% of the variance while drastically reducing training time and model complexity.
* **Leakage Prevention:** Engineered the entire workflow inside Scikit-Learn `Pipeline` objects. This strictly isolated the training folds from validation data during feature scaling and PCA decomposition, ensuring realistic, non-overfitted performance metrics.
* **Feature Value Insights:** Hyperparameter tuning via `GridSearchCV` revealed that regularization constants ($\alpha$) needed strict calibration to balance the high variance introduced by engineered interaction terms.
