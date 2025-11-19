import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

data_train_bruised_path = 'data/features/bruised/train_features.csv'
data_test_bruised_path = 'data/features/bruised/test_features.csv'

data_train_processed_bruised_path = 'data/features/processed_bruised/train_data_processed.csv'
data_test_processed_bruised_path = 'data/features/processed_bruised/test_data_processed.csv'

train_data = pd.read_csv(data_train_bruised_path)
test_data = pd.read_csv(data_test_bruised_path)

def scale_bruised_data(train_df, test_df):
    numeric_cols = [col for col in train_df.columns if col not in ['apple_id', 'damaged']]
    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    return train_df, test_df

def save_processed_bruised_data(train_df, test_df, train_path, test_path):
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


train_data, test_data = scale_bruised_data(train_data, test_data)
save_processed_bruised_data(train_data, test_data, data_train_processed_bruised_path, data_test_processed_bruised_path)