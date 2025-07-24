import pytest
import pandas as pd
from src.data_processing import add_rfm_high_risk, prepare_features_target

def test_add_rfm_high_risk_adds_column():
    """Test if 'is_high_risk' column is added by add_rfm_high_risk."""
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2, 2, 3, 3],  # added customer 3
        'TransactionStartTime': pd.to_datetime([
            '2023-07-20', '2023-07-21', 
            '2023-07-22', '2023-07-23',
            '2023-07-24', '2023-07-25'
        ]),
        'TransactionId': [101, 102, 201, 202, 301, 302],
        'Amount': [100, 150, 200, 250, 300, 350]
    })

    result_df = add_rfm_high_risk(df)

    assert 'is_high_risk' in result_df.columns, "'is_high_risk' column not found"
    assert pd.api.types.is_integer_dtype(result_df['is_high_risk']), "'is_high_risk' is not an integer type"

def test_prepare_features_target_drops_columns():
    """Test if prepare_features_target drops the correct columns and returns expected features and target."""
    df = pd.DataFrame({
        'FraudResult': [0, 1],
        'TransactionId': [101, 102],
        'BatchId': [1, 1],
        'AccountId': [10, 11],
        'SubscriptionId': [5, 5],
        'CustomerId': [1, 2],
        'Feature1': [0.5, 0.6]
    })

    X, y = prepare_features_target(df)

    dropped_cols = ['FraudResult', 'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
    for col in dropped_cols:
        assert col not in X.columns, f"{col} should have been dropped from features"

    assert 'Feature1' in X.columns, "'Feature1' missing in features"
    assert y.equals(df['FraudResult']), "Target variable 'y' does not match expected values"
