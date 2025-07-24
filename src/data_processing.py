# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

#Calculate RFM and assign high-risk cluster ---
def add_rfm_high_risk(df, snapshot_date=None, random_state=42):
    df = df.copy()
    # Ensure TransactionStartTime is datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Define snapshot_date for Recency calculation (default to max date in data + 1 day)
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    # Calculate RFM metrics grouped by CustomerId
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
        'TransactionId': 'count',                                         # Frequency
        'Amount': 'sum'                                                  # Monetary
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })

    # Handle any negative or zero monetary values by replacing with small positive (if needed)
    rfm['Monetary'] = rfm['Monetary'].clip(lower=0.01)

    # Scale RFM features before clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # KMeans clustering into 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    # Identify the cluster with highest risk: usually the cluster with highest Recency (most days since last purchase), and lowest Frequency & Monetary
    cluster_summary = rfm.groupby('cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })

    # The "high risk" cluster is the one with highest Recency and lowest Frequency & Monetary.
    # Define a score to identify: higher Recency + lower Frequency + lower Monetary = higher risk
    cluster_summary['risk_score'] = cluster_summary['Recency'] - cluster_summary['Frequency'] - cluster_summary['Monetary']
    high_risk_cluster = cluster_summary['risk_score'].idxmax()

    # Create binary is_high_risk label: 1 if in high risk cluster else 0
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

    # Merge back is_high_risk label to original dataframe by CustomerId
    df = df.merge(rfm['is_high_risk'], left_on='CustomerId', right_index=True, how='left')

    print(f"[add_rfm_high_risk] Assigned high risk cluster: {high_risk_cluster}")
    print(f"[add_rfm_high_risk] High risk customers count: {rfm['is_high_risk'].sum()}")

    return df

# --- 6. Full processing function ---
def process_and_save(input_raw_path, output_X_path, output_y_path, pipeline_path=None):
    # Load raw data
    df = load_data(input_raw_path)

    # --- TASK 4: Add proxy target variable is_high_risk ---
    df = add_rfm_high_risk(df)

    # Use is_high_risk as new target instead of FraudResult for credit risk modeling
    y = df['is_high_risk']

    # Determine numeric and categorical columns from raw df BEFORE transformations
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Remove columns we won't use as features
    for col in ['is_high_risk', 'FraudResult', 'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime']:
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

    # Save processed features and new target
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
