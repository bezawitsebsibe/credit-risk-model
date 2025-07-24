import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def data_overview(df):
    print(df.info())
    print(df.isnull().sum())
    print("Duplicates:", df.duplicated().sum())

def preprocess_datetime(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['hour'] = df['TransactionStartTime'].dt.hour
    df['dayofweek'] = df['TransactionStartTime'].dt.dayofweek
    df['day'] = df['TransactionStartTime'].dt.day
    df['month'] = df['TransactionStartTime'].dt.month
    return df

def prepare_features_target(df):
    y = df['FraudResult']
    X = df.drop(columns=[
        'FraudResult',
        'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
        'TransactionStartTime'
    ])
    return X, y

def save_processed_data(X, y, output_X_path, output_y_path):
    X.to_csv(output_X_path, index=False)
    y.to_csv(output_y_path, index=False)

def process_pipeline(input_raw_path, output_X_path, output_y_path):
    df = load_data(input_raw_path)
    data_overview(df)  # optional, can comment out later for cleaner runs
    df = preprocess_datetime(df)
    X, y = prepare_features_target(df)
    save_processed_data(X, y, output_X_path, output_y_path)
    print(f"Processed data saved to:\n{output_X_path}\n{output_y_path}")

if __name__ == "__main__":
    process_pipeline(
        input_raw_path='data/raw/data.csv',
        output_X_path='data/processed/X_final.csv', #features only
        output_y_path='data/processed/y_final.csv' #target only
    )
