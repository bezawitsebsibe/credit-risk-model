# credit-risk-model

## Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretable Models

The Basel II Accord is like a rulebook for banks. It tells them to measure credit risk carefully and explain their decisions. This means we can't just use a model that gives a "yes" or "no" answer without saying why. We need a model that is explainable, documented, and easy to understand — especially in financial environments. This is why banks often use models like Logistic Regression with Weight of Evidence (WoE), which are trusted and transparent.

### 2. The Importance of Proxy Variables

In our dataset, we don’t know exactly who failed to pay back a loan (there’s no "default" column). So, we create a proxy variable — a label that guesses which customers are high-risk based on their behavior. This is necessary to train the model. But if we define this label badly, the model might make poor predictions — such as denying good customers or giving loans to risky ones. That’s why it’s important to create the proxy with care, using methods like RFM (Recency, Frequency, Monetary).

###Task 2: Exploratory Data Analysis (EDA)

- Explored dataset structure including size, feature types, and data quality
- Identified minimal missing values, enabling straightforward preprocessing
- Visualized distributions of numerical features showing skewness and outliers
- Analyzed categorical feature frequencies and variability
- Discovered moderate correlations between numerical features useful for modeling
- Summarized key insights to guide further feature engineering and model building

### 3. Trade-offs Between Simple and Complex Models

Simple models like Logistic Regression are easy to explain and accepted in banking, especially when using WoE encoding. But they may not capture complex patterns. Complex models like Gradient Boosting Machines (GBM) are more powerful and accurate but harder to explain. In regulated industries like banking, we must choose models that not only perform well but are also transparent and trustworthy. Often, a balance between interpretability and performance is needed.

Task 3: Implement automated feature engineering pipeline

- Created custom transformers for datetime feature extraction and customer-level aggregation
- Built full sklearn Pipeline combining preprocessing of numerical and categorical features
- Added imputation, scaling, and one-hot encoding for clean feature processing
- Output processed feature matrix and target for model training
- Saved pipeline artifact for future inference and deployment


### Task 4 - Proxy Target Variable Engineering
- Created an RFM-based high-risk proxy label (`is_high_risk`) using transaction data.
- Implemented feature engineering pipeline with datetime and aggregate features.
- Added unit tests to verify data processing.

### Task 5 - Model Training and Tracking
- Developed training script with train/test split and model selection (Logistic Regression and Random Forest).
- Included hyperparameter tuning with GridSearchCV.
- Logged models, parameters, and metrics using MLflow.
- Registered best model in MLflow Model Registry.
- Added unit tests for helper functions.
- CI pipeline runs linter and tests automatically.

### Task 6 - Model Deployment and Continuous Integration
- Created a REST API using FastAPI to serve model predictions.
- Defined request/response schemas using Pydantic models.
- Containerized API with Docker and docker-compose.
- Set up GitHub Actions workflow to run linter and tests on each push.
- Included requirements update with FastAPI, Uvicorn, MLflow, and linter.