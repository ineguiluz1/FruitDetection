import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold

train_data_path = 'data/augmented/features/raw/train_features.csv'
test_data_path = 'data/augmented/features/raw/test_features.csv'

train_processed_path = 'data/augmented/features/processed/train_features_processed.csv'
test_processed_path = 'data/augmented/features/processed/test_features_processed.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

def simplify_labels(df, label_col='label'):
    df[label_col] = df[label_col].str.split().str[0]
    return df

def scale_data(train_df, test_df):
    numeric_cols = [col for col in train_df.columns if col != 'label']
    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    return train_df, test_df

def encode_labels(train_df, test_df, label_col='label'):
    encoder = LabelEncoder()
    train_df['label_encoded'] = encoder.fit_transform(train_df[label_col])
    test_df['label_encoded'] = encoder.transform(test_df[label_col])
    return train_df, test_df, encoder

def save_processed_data(train_df, test_df, train_path, test_path):
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

def balance_dataset(train_df, label_col='label'):
    median_samples = int(train_df[label_col].value_counts().median())
    balanced_df = train_df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(min(len(x), median_samples), random_state=42)
    ).reset_index(drop=True)
    return balanced_df

train_data = simplify_labels(train_data)
test_data = simplify_labels(test_data)
train_data, test_data = scale_data(train_data, test_data)
train_data = balance_dataset(train_data)
train_data, test_data, label_encoder = encode_labels(train_data, test_data, label_col='label')
save_processed_data(train_data, test_data, train_processed_path, test_processed_path)




