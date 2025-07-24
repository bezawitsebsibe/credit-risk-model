# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# --- 1. Load data (simple helper function) ---
def load_data(path):
    return pd.read_csv(path)

# --- 2. Custom transformer: Extract datetime features ---
class DateTimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Convert to datetime if not already
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        
        # Extract features
        X['hour'] = X[self.datetime_col].dt.hour
        X['dayofweek'] = X[self.datetime_col].dt.dayofweek
        X['day'] = X[self.datetime_col].dt.day
        X['month'] = X[self.datetime_col].dt.month
        X['year'] = X[self.datetime_col].dt.year
        
        # Drop original datetime to avoid duplication
        X = X.drop(columns=[self.datetime_col])

        print(f"[DateTimeFeaturesExtractor] Output shape: {X.shape}")
        print(f"[DateTimeFeaturesExtractor] Columns: {X.columns.tolist()}")

        return X

# --- 3. Custom transformer: Aggregate features example ---
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='CustomerId', agg_col='Amount'):
        self.group_col = group_col
        self.agg_col = agg_col
        
    def fit(self, X, y=None):
        # Precompute aggregate statistics
        self.agg_sum_ = X.groupby(self.group_col)[self.agg_col].sum()
        self.agg_mean_ = X.groupby(self.group_col)[self.agg_col].mean()
        self.agg_count_ = X.groupby(self.group_col)[self.agg_col].count()
        self.agg_std_ = X.groupby(self.group_col)[self.agg_col].std().fillna(0)
        return self
    
    def transform(self, X):
        X = X.copy()
        # Map aggregated features back to rows by CustomerId
        X['total_amount'] = X[self.group_col].map(self.agg_sum_)
        X['avg_amount'] = X[self.group_col].map(self.agg_mean_)
        X['transaction_count'] = X[self.group_col].map(self.agg_count_)
        X['std_amount'] = X[self.group_col].map(self.agg_std_)
        
        # IMPORTANT: Do NOT drop group_col here, keep it for next step to avoid shape mismatch
        # X = X.drop(columns=[self.group_col])
        
        print(f"[AggregateFeatures] Output shape: {X.shape}")
        print(f"[AggregateFeatures] Columns: {X.columns.tolist()}")
        return X

# --- 4. Prepare feature / target split and drop columns not needed ---
def prepare_features_target(df, target_col='FraudResult'):
    y = df[target_col]
    
    # Drop columns that leak info or IDs that won't help the model
    drop_cols = [
        target_col,
        'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId'
    ]
    X = df.drop(columns=drop_cols)
    return X, y

# --- 5. Build the full preprocessing pipeline ---
def build_preprocessing_pipeline(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # fixed here
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    full_pipeline = Pipeline(steps=[
        ('datetime', DateTimeFeaturesExtractor(datetime_col='TransactionStartTime')),
        ('aggregate', AggregateFeatures(group_col='CustomerId', agg_col='Amount')),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline


# --- 6. Full processing function ---
def process_and_save(input_raw_path, output_X_path, output_y_path, pipeline_path=None):
    # Load raw data
    df = load_data(input_raw_path)

    # Prepare target variable
    y = df['FraudResult']

    # Determine numeric and categorical columns from raw df BEFORE transformations
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Remove columns we won't use as features
    for col in ['FraudResult', 'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime']:
        if col in numeric_cols:
            numeric_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    # Build full pipeline with these columns
    pipeline = build_preprocessing_pipeline(numeric_cols, categorical_cols)

    # Fit transform full pipeline on raw df
    X_processed = pipeline.fit_transform(df)

    print(f"Full pipeline output shape: {X_processed.shape}")

    # Get feature names from ColumnTransformer
    num_features = numeric_cols
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_features = ohe.get_feature_names_out(categorical_cols).tolist()

    feature_names = num_features + cat_features

    # Create DataFrame from numpy array (dense now, so no problem)
    X_df = pd.DataFrame(X_processed, columns=feature_names)

    # Save processed features and target
    X_df.to_csv(output_X_path, index=False)
    y.to_csv(output_y_path, index=False)

    # Save pipeline if needed
    if pipeline_path:
        joblib.dump(pipeline, pipeline_path)

    print(f"Processed data saved to:\n{output_X_path}\n{output_y_path}")
    if pipeline_path:
        print(f"Pipeline saved to: {pipeline_path}")

# --- If run as script ---
if __name__ == "__main__":
    process_and_save(
        input_raw_path='data/raw/data.csv',
        output_X_path='data/processed/X_final.csv',
        output_y_path='data/processed/y_final.csv',
        pipeline_path='models/preprocessing_pipeline.pkl'
    )
