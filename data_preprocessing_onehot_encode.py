"""
NSL-KDD Dataset Preprocessing with One-Hot Encoding
==================================================

This script implements data preprocessing for the NSL-KDD dataset using one-hot encoding
for categorical features, following the algorithm described in the research paper.

Processing Steps:
1. Feature Selection - Remove low information gain features
2. Categorical Encoding - One-hot encoding for categorical features
3. Data Normalization - Log scaling and value mapping
4. Binary Classification - Convert multi-class to binary (normal/attack)

Author: Doo-Seop Choi
Date: 2025.05.30
License: MIT
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
import warnings

warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'input_file': './data/NSL-KDD-All(Train+Test).csv',
    'output_dir': './preprocessed_data',
    'random_state': 42
}

# Feature categories based on NSL-KDD dataset
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
LOW_INFORMATION_FEATURES = ['is_host_login', 'is_guest_login', 'num_outbound_cmds', 'num_shells', 'urgent']
LOG_SCALING_FEATURES = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins',
                        'num_compromised', 'num_root', 'num_file_creations', 'num_access_files',
                        'count', 'srv_count']
BINARY_FEATURES = ['logged_in', 'root_shell']
DISCRETE_FEATURES = ['su_attempted']  # Values: 0, 1, 2
RATE_FEATURES = ['serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate']


def load_dataset(file_path):
    """
    Load NSL-KDD dataset from CSV file.

    Args:
        file_path (str): Path to the NSL-KDD CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✓ Dataset loaded successfully")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {len(df.columns)}")

    # Display class distribution
    if 'classification.' in df.columns:
        print(f"  - Class distribution:")
        for class_name, count in df['classification.'].value_counts().items():
            print(f"    {class_name}: {count}")

    return df


def remove_low_information_features(df, features_to_remove):
    """
    Remove features with low information gain.

    Args:
        df (pd.DataFrame): Input dataframe
        features_to_remove (list): List of feature names to remove

    Returns:
        pd.DataFrame: Dataframe with features removed
    """
    print(f"Removing {len(features_to_remove)} low information gain features...")

    # Check which features actually exist in the dataframe
    existing_features = [f for f in features_to_remove if f in df.columns]
    missing_features = [f for f in features_to_remove if f not in df.columns]

    if missing_features:
        print(f"  Warning: Features not found in dataset: {missing_features}")

    if existing_features:
        df_reduced = df.drop(columns=existing_features)
        print(f"  ✓ Removed features: {existing_features}")
        print(f"  - Remaining features: {len(df_reduced.columns)}")
    else:
        df_reduced = df.copy()
        print(f"  No features were removed")

    return df_reduced


def encode_categorical_features(df, categorical_features):
    """
    Apply one-hot encoding to categorical features.

    Args:
        df (pd.DataFrame): Input dataframe
        categorical_features (list): List of categorical feature names

    Returns:
        pd.DataFrame: Dataframe with one-hot encoded features
    """
    print("Applying one-hot encoding to categorical features...")

    # Check which categorical features exist
    existing_categorical = [f for f in categorical_features if f in df.columns]
    missing_categorical = [f for f in categorical_features if f not in df.columns]

    if missing_categorical:
        print(f"  Warning: Categorical features not found: {missing_categorical}")

    if not existing_categorical:
        print("  No categorical features found to encode")
        return df

    # Extract categorical data
    categorical_data = df[existing_categorical].copy()

    # Generate column names for one-hot encoding
    dummy_columns = []
    for feature in existing_categorical:
        unique_values = sorted(df[feature].unique())
        feature_columns = [f"{feature}_{value}" for value in unique_values]
        dummy_columns.extend(feature_columns)

    print(f"  - Features to encode: {existing_categorical}")
    print(f"  - Total dummy columns to create: {len(dummy_columns)}")

    # Apply label encoding first
    categorical_encoded = categorical_data.apply(LabelEncoder().fit_transform)

    # Apply one-hot encoding
    encoder = OneHotEncoder(sparse_output=False)
    categorical_onehot = encoder.fit_transform(categorical_encoded)

    # Create dataframe with proper column names
    categorical_df = pd.DataFrame(categorical_onehot, columns=dummy_columns)

    # Join with original dataframe (excluding original categorical columns)
    df_encoded = df.drop(columns=existing_categorical).join(categorical_df)

    print(f"  ✓ One-hot encoding completed")
    print(f"  - Original categorical features removed: {existing_categorical}")
    print(f"  - New dummy features added: {len(dummy_columns)}")
    print(f"  - Final shape: {df_encoded.shape}")

    return df_encoded


def apply_log_scaling(df, log_features):
    """
    Apply log10 scaling to specified features.

    Args:
        df (pd.DataFrame): Input dataframe
        log_features (list): List of features to apply log scaling

    Returns:
        pd.DataFrame: Dataframe with log scaled features
    """
    print("Applying log scaling to numerical features...")

    df_scaled = df.copy()
    existing_log_features = [f for f in log_features if f in df.columns]
    missing_log_features = [f for f in log_features if f not in df.columns]

    if missing_log_features:
        print(f"  Warning: Log scaling features not found: {missing_log_features}")

    if existing_log_features:
        for feature in existing_log_features:
            # Apply log10(x + 1) to handle zero values
            df_scaled[feature] = np.log10(df_scaled[feature] + 1)

        print(f"  ✓ Log scaling applied to: {existing_log_features}")
    else:
        print("  No features found for log scaling")

    return df_scaled


def apply_binary_mapping(df, binary_features):
    """
    Map binary features: 0 -> 0, 1 -> 255.

    Args:
        df (pd.DataFrame): Input dataframe
        binary_features (list): List of binary features

    Returns:
        pd.DataFrame: Dataframe with binary mapping applied
    """
    print("Applying binary mapping (0->0, 1->255)...")

    df_mapped = df.copy()
    existing_binary = [f for f in binary_features if f in df.columns]
    missing_binary = [f for f in binary_features if f not in df.columns]

    if missing_binary:
        print(f"  Warning: Binary features not found: {missing_binary}")

    if existing_binary:
        for feature in existing_binary:
            df_mapped.loc[df_mapped[feature] == 1, feature] = 255
            # 0 values remain 0

        print(f"  ✓ Binary mapping applied to: {existing_binary}")

        # Display value counts for verification
        for feature in existing_binary:
            print(f"    {feature} values: {df_mapped[feature].value_counts().to_dict()}")
    else:
        print("  No binary features found for mapping")

    return df_mapped


def apply_discrete_mapping(df, discrete_features):
    """
    Map discrete features: 0->85, 1->170, 2->255.

    Args:
        df (pd.DataFrame): Input dataframe
        discrete_features (list): List of discrete features

    Returns:
        pd.DataFrame: Dataframe with discrete mapping applied
    """
    print("Applying discrete mapping (0->85, 1->170, 2->255)...")

    df_mapped = df.copy()
    existing_discrete = [f for f in discrete_features if f in df.columns]
    missing_discrete = [f for f in discrete_features if f not in df.columns]

    if missing_discrete:
        print(f"  Warning: Discrete features not found: {missing_discrete}")

    if existing_discrete:
        for feature in existing_discrete:
            df_mapped.loc[df_mapped[feature] == 0, feature] = 85
            df_mapped.loc[df_mapped[feature] == 1, feature] = 170
            df_mapped.loc[df_mapped[feature] == 2, feature] = 255

        print(f"  ✓ Discrete mapping applied to: {existing_discrete}")

        # Display value counts for verification
        for feature in existing_discrete:
            print(f"    {feature} values: {df_mapped[feature].value_counts().to_dict()}")
    else:
        print("  No discrete features found for mapping")

    return df_mapped


def apply_rate_scaling(df, rate_features):
    """
    Scale rate features by multiplying with 255.

    Args:
        df (pd.DataFrame): Input dataframe
        rate_features (list): List of rate features

    Returns:
        pd.DataFrame: Dataframe with rate scaling applied
    """
    print("Applying rate scaling (multiply by 255)...")

    df_scaled = df.copy()
    existing_rate = [f for f in rate_features if f in df.columns]
    missing_rate = [f for f in rate_features if f not in df.columns]

    if missing_rate:
        print(f"  Warning: Rate features not found: {missing_rate}")

    if existing_rate:
        for feature in existing_rate:
            df_scaled[feature] = df_scaled[feature] * 255

        print(f"  ✓ Rate scaling applied to: {existing_rate}")
    else:
        print("  No rate features found for scaling")

    return df_scaled


def convert_to_binary_classification(df, target_column='classification.'):
    """
    Convert multi-class classification to binary (normal vs attack).

    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column

    Returns:
        pd.DataFrame: Dataframe with binary classification
    """
    print("Converting to binary classification...")

    if target_column not in df.columns:
        print(f"  Warning: Target column '{target_column}' not found")
        return df

    df_binary = df.copy()

    # Display original class distribution
    print(f"  Original class distribution:")
    for class_name, count in df_binary[target_column].value_counts().items():
        print(f"    {class_name}: {count}")

    # Convert attack types to 'attack'
    attack_types = ['Dos', 'Probe', 'R2L', 'U2R']
    for attack_type in attack_types:
        df_binary.loc[df_binary[target_column] == attack_type, target_column] = 'attack'

    # Display new class distribution
    print(f"  ✓ Binary classification applied")
    print(f"  New class distribution:")
    for class_name, count in df_binary[target_column].value_counts().items():
        print(f"    {class_name}: {count}")

    return df_binary


def convert_data_types(df):
    """
    Convert float64 columns to int64 for consistency.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with converted data types
    """
    print("Converting data types...")

    df_converted = df.copy()
    float_columns = []

    for column in df_converted.columns:
        if df_converted[column].dtype == 'float64':
            # Check if all values are actually integers
            if df_converted[column].notna().all() and (df_converted[column] % 1 == 0).all():
                df_converted[column] = df_converted[column].astype('int64')
                float_columns.append(column)

    if float_columns:
        print(f"  ✓ Converted {len(float_columns)} columns from float64 to int64")
    else:
        print("  No columns needed conversion")

    return df_converted


def reorder_columns(df, target_column='classification.'):
    """
    Reorder columns to place one-hot encoded features at the end.

    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column

    Returns:
        pd.DataFrame: Dataframe with reordered columns
    """
    print("Reordering columns...")

    if target_column not in df.columns:
        print(f"  Warning: Target column '{target_column}' not found")
        return df

    # Get column lists
    columns = df.columns.tolist()

    # Find the index of the target column
    target_index = columns.index(target_column)

    # Separate columns: target, original features, one-hot encoded features
    target_col = [target_column]

    # Original features (before one-hot encoding)
    original_features = []
    onehot_features = []

    for col in columns:
        if col != target_column:
            # Check if it's a one-hot encoded feature (contains '_' and follows pattern)
            if any(col.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES):
                onehot_features.append(col)
            else:
                original_features.append(col)

    # Reorder: target + one-hot + original
    new_order = target_col + onehot_features + original_features

    df_reordered = df[new_order]

    print(f"  ✓ Columns reordered")
    print(f"    - Target column: 1")
    print(f"    - One-hot encoded features: {len(onehot_features)}")
    print(f"    - Original features: {len(original_features)}")

    return df_reordered


def save_preprocessed_data(df, output_dir, filename_prefix="onehot_preprocessed"):
    """
    Save preprocessed data to CSV files.

    Args:
        df (pd.DataFrame): Preprocessed dataframe
        output_dir (str): Output directory
        filename_prefix (str): Prefix for output filenames
    """
    print("Saving preprocessed data...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save complete dataset
    complete_file = os.path.join(output_dir, f"{filename_prefix}_nslkdd.csv")
    df.to_csv(complete_file, index=False)
    print(f"  ✓ Complete dataset saved: {complete_file}")

    # Save modified column order version
    modified_file = os.path.join(output_dir, f"{filename_prefix}_nslkdd_modified_columns.csv")
    df_reordered = reorder_columns(df)
    df_reordered.to_csv(modified_file, index=False)
    print(f"  ✓ Modified columns dataset saved: {modified_file}")

    # Separate and save normal/attack data
    if 'classification.' in df.columns:
        normal_df = df[df['classification.'] == 'normal']
        attack_df = df[df['classification.'] == 'attack']

        normal_file = os.path.join(output_dir, f"{filename_prefix}_nslkdd_normal.csv")
        attack_file = os.path.join(output_dir, f"{filename_prefix}_nslkdd_attack.csv")

        normal_df.to_csv(normal_file, index=False)
        attack_df.to_csv(attack_file, index=False)

        print(f"  ✓ Normal data saved: {normal_file} ({len(normal_df)} samples)")
        print(f"  ✓ Attack data saved: {attack_file} ({len(attack_df)} samples)")


def main():
    """
    Main function to run the complete one-hot encoding preprocessing pipeline.
    """
    print("=" * 60)
    print("NSL-KDD Dataset Preprocessing with One-Hot Encoding")
    print("=" * 60)

    try:
        # Step 1: Load dataset
        df = load_dataset(CONFIG['input_file'])

        # Step 2: Remove low information gain features
        df = remove_low_information_features(df, LOW_INFORMATION_FEATURES)

        # Step 3: Apply one-hot encoding to categorical features
        df = encode_categorical_features(df, CATEGORICAL_FEATURES)

        # Step 4: Apply log scaling
        df = apply_log_scaling(df, LOG_SCALING_FEATURES)

        # Step 5: Apply binary mapping
        df = apply_binary_mapping(df, BINARY_FEATURES)

        # Step 6: Apply discrete mapping
        df = apply_discrete_mapping(df, DISCRETE_FEATURES)

        # Step 7: Apply rate scaling
        df = apply_rate_scaling(df, RATE_FEATURES)

        # Step 8: Convert to binary classification
        df = convert_to_binary_classification(df)

        # Step 9: Convert data types
        df = convert_data_types(df)

        # Step 10: Save preprocessed data
        save_preprocessed_data(df, CONFIG['output_dir'], "onehot_preprocessed")

        print("\n" + "=" * 60)
        print("✓ One-hot encoding preprocessing completed successfully!")
        print(f"✓ Final dataset shape: {df.shape}")
        print(f"✓ Output saved to: {CONFIG['output_dir']}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()