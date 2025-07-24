import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

# Load processed data
def load_processed_data(X_path='data/processed/X_final.csv', y_path='data/processed/y_final.csv'):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()  # Convert to Series
    return X, y

# Split data into train and test sets
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Evaluate model metrics on test set
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    return metrics

# Train logistic regression model
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

# Tune and train random forest with GridSearchCV
def train_tuned_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# MLflow training and logging wrapper
def train_and_log_model(name, model, X_train, y_train, X_test, y_test, params=None):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path='model')

        print(f"[{name}] Metrics: {metrics}")

        return model, metrics

def main():
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train Logistic Regression
    logreg = train_logistic_regression(X_train, y_train)
    train_and_log_model('Logistic_Regression', logreg, X_train, y_train, X_test, y_test)

    # Train Random Forest with hyperparameter tuning
    rf_model, best_params = train_tuned_random_forest(X_train, y_train)
    train_and_log_model('Random_Forest', rf_model, X_train, y_train, X_test, y_test, best_params)

if __name__ == '__main__':
    main()
