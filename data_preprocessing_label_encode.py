"""
NSL-KDD Dataset Preprocessing with Automated Label Encoding
==========================================================

This script implements data preprocessing for the NSL-KDD dataset using scikit-learn's
LabelEncoder for categorical features, while maintaining the specific numeric mappings
described in the research paper.

Processing Steps:
1. Feature Selection - Remove low information gain features
2. Data Transformation - Log scaling for numerical features
3. Categorical Encoding - Automated label encoding with paper-specific mappings
4. Value Mapping - Binary and discrete feature mapping
5. Normalization - Scale features to [0, 255] range
6. Binary Classification - Convert multi-class to binary (normal/attack)

Author: Doo-Seop Choi
Date: 2025.05.30
License: MIT
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'input_file': './data/NSL-KDD-All(Train+Test).csv',
    'output_dir': './preprocessed_data',
    'random_state': 42
}

# Feature categories based on NSL-KDD dataset and algorithm
LOW_INFORMATION_FEATURES = ['is_host_login', 'is_guest_login', 'num_outbound_cmds', 'num_shells', 'urgent']
LOG_SCALING_FEATURES = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins',
                        'num_compromised', 'num_root', 'num_file_creations', 'num_access_files',
                        'count', 'srv_count']
BINARY_FEATURES = ['logged_in', 'root_shell']
DISCRETE_FEATURES = ['su_attempted']  # Values: 0, 1, 2
RATE_FEATURES = ['serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate']

# Paper-specific mappings for consistency with research
PAPER_MAPPINGS = {
    'protocol_type': {
        'tcp': 85,
        'udp': 170,
        'icmp': 255
    },
    'service': {
        'ftp_data': 3, 'other': 6, 'private': 9, 'http': 12, 'remote_job': 15, 'name': 18, 'netbios_ns': 21,
        'eco_i': 24, 'mtp': 27, 'telnet': 30, 'finger': 33, 'domain_u': 36, 'supdup': 39, 'uucp_path': 42,
        'Z39_50': 45, 'smtp': 48, 'csnet_ns': 51, 'uucp': 54, 'netbios_dgm': 57, 'urp_i': 60, 'auth': 63,
        'domain': 66, 'ftp': 69, 'bgp': 72, 'ldap': 75, 'ecr_i': 78, 'gopher': 81, 'vmnet': 84,
        'systat': 87, 'http_443': 90, 'efs': 93, 'whois': 96, 'imap4': 99, 'iso_tsap': 102,
        'echo': 105, 'klogin': 108, 'link': 111, 'sunrpc': 114, 'login': 117, 'kshell': 120, 'sql_net': 123,
        'time': 126, 'hostnames': 129, 'exec': 132, 'ntp_u': 135, 'discard': 138, 'nntp': 141, 'courier': 144,
        'ctf': 147, 'ssh': 150, 'daytime': 153, 'shell': 156, 'netstat': 159, 'pop_3': 162, 'nnsp': 165,
        'IRC': 168, 'pop_2': 171, 'printer': 174, 'tim_i': 177, 'pm_dump': 180, 'red_i': 183, 'netbios_ssn': 186,
        'rje': 189, 'X11': 192, 'urh_i': 195, 'http_8001': 198, 'aol': 201, 'http_2784': 204, 'tftp_u': 207, 'harvest': 210
    },
    'flag': {
        'SF': 23, 'S0': 46, 'REJ': 69, 'RSTR': 92, 'SH': 115, 'RSTO': 138,
        'S1': 161, 'RSTOS0': 184, 'S3': 207, 'S2': 230, 'OTH': 253
    }
}


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
    Remove features with low information gain based on algorithm.

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


def apply_log_scaling(df, log_features):
    """
    Apply log10 scaling to specified numerical features.

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
            original_min = df_scaled[feature].min()
            original_max = df_scaled[feature].max()
            df_scaled[feature] = np.log10(df_scaled[feature] + 1)
            new_min = df_scaled[feature].min()
            new_max = df_scaled[feature].max()
            print(f"    {feature}: [{original_min:.2f}, {original_max:.2f}] -> [{new_min:.4f}, {new_max:.4f}]")

        print(f"  ✓ Log scaling applied to {len(existing_log_features)} features")
    else:
        print("  No features found for log scaling")

    return df_scaled


def create_paper_compatible_encoder(feature_name, unique_values, paper_mapping):
    """
    Create a LabelEncoder that produces mappings compatible with the paper's specifications.

    Args:
        feature_name (str): Name of the feature being encoded
        unique_values (array): Unique values found in the dataset
        paper_mapping (dict): Paper-specific mapping dictionary

    Returns:
        LabelEncoder: Configured encoder
        dict: Actual mapping used
    """
    encoder = LabelEncoder()

    # Check if all unique values exist in paper mapping
    missing_values = set(unique_values) - set(paper_mapping.keys())
    if missing_values:
        print(f"    Warning: Values not in paper mapping for {feature_name}: {missing_values}")
        # Create mapping for missing values
        max_paper_value = max(paper_mapping.values()) if paper_mapping else 0
        for i, missing_val in enumerate(sorted(missing_values)):
            paper_mapping[missing_val] = max_paper_value + (i + 1) * 10

    # Create ordered list based on paper mapping values
    sorted_values = sorted(unique_values, key=lambda x: paper_mapping.get(x, float('inf')))

    # Fit encoder with sorted values to ensure consistent ordering
    encoder.fit(sorted_values)

    # Create the actual mapping that will be applied
    actual_mapping = {}
    for value in unique_values:
        if value in paper_mapping:
            actual_mapping[value] = paper_mapping[value]
        else:
            # Fallback for unmapped values
            encoded_label = encoder.transform([value])[0]
            actual_mapping[value] = encoded_label

    return encoder, actual_mapping


def apply_automated_categorical_encoding(df):
    """
    Apply automated label encoding to categorical features using scikit-learn's LabelEncoder
    while maintaining paper-specific mappings.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with encoded categorical features
    """
    print("Applying automated categorical encoding with paper-specific mappings...")

    df_encoded = df.copy()
    encoding_summary = {}

    # Protocol type encoding
    if 'protocol_type' in df_encoded.columns:
        print("  Encoding protocol_type...")
        unique_protocols = df_encoded['protocol_type'].unique()
        print(f"    Unique protocols: {sorted(unique_protocols)}")

        encoder, mapping = create_paper_compatible_encoder(
            'protocol_type', unique_protocols, PAPER_MAPPINGS['protocol_type']
        )

        # Apply paper-specific mapping directly
        df_encoded['protocol_type'] = df_encoded['protocol_type'].map(mapping)
        encoding_summary['protocol_type'] = mapping

        # Check for unmapped values
        unmapped = df_encoded['protocol_type'].isna().sum()
        if unmapped > 0:
            print(f"    Warning: {unmapped} unmapped protocol_type values")

        print(f"    ✓ Protocol mapping applied: {mapping}")
        print(f"    Value distribution: {df_encoded['protocol_type'].value_counts().to_dict()}")

    # Service encoding
    if 'service' in df_encoded.columns:
        print("  Encoding service...")
        unique_services = df_encoded['service'].unique()
        print(f"    Unique services: {len(unique_services)}")

        encoder, mapping = create_paper_compatible_encoder(
            'service', unique_services, PAPER_MAPPINGS['service']
        )

        # Apply paper-specific mapping directly
        df_encoded['service'] = df_encoded['service'].map(mapping)
        encoding_summary['service'] = mapping

        # Check for unmapped values
        unmapped = df_encoded['service'].isna().sum()
        if unmapped > 0:
            print(f"    Warning: {unmapped} unmapped service values, filling with default value 1")
            df_encoded['service'].fillna(1, inplace=True)

        print(f"    ✓ Service mapping applied ({len(mapping)} mappings)")

    # Flag encoding
    if 'flag' in df_encoded.columns:
        print("  Encoding flag...")
        unique_flags = df_encoded['flag'].unique()
        print(f"    Unique flags: {sorted(unique_flags)}")

        encoder, mapping = create_paper_compatible_encoder(
            'flag', unique_flags, PAPER_MAPPINGS['flag']
        )

        # Apply paper-specific mapping directly
        df_encoded['flag'] = df_encoded['flag'].map(mapping)
        encoding_summary['flag'] = mapping

        # Check for unmapped values
        unmapped = df_encoded['flag'].isna().sum()
        if unmapped > 0:
            print(f"    Warning: {unmapped} unmapped flag values")

        print(f"    ✓ Flag mapping applied: {mapping}")
        print(f"    Value distribution: {df_encoded['flag'].value_counts().to_dict()}")

    print("  ✓ Automated categorical encoding completed")

    # Save encoding summary for reference
    save_encoding_summary(encoding_summary)

    return df_encoded


def save_encoding_summary(encoding_summary):
    """
    Save the encoding summary to a file for reference.

    Args:
        encoding_summary (dict): Dictionary containing encoding mappings
    """
    output_file = os.path.join(CONFIG['output_dir'], 'encoding_summary.txt')
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("Categorical Feature Encoding Summary\n")
        f.write("=" * 50 + "\n\n")

        for feature_name, mapping in encoding_summary.items():
            f.write(f"{feature_name.upper()}:\n")
            f.write("-" * 20 + "\n")
            for original, encoded in sorted(mapping.items(), key=lambda x: x[1]):
                f.write(f"  {original} -> {encoded}\n")
            f.write("\n")

    print(f"  ✓ Encoding summary saved: {output_file}")


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
            original_dist = df_mapped[feature].value_counts().to_dict()
            df_mapped.loc[df_mapped[feature] == 1, feature] = 255
            # 0 values remain 0
            new_dist = df_mapped[feature].value_counts().to_dict()

            print(f"    {feature}: {original_dist} -> {new_dist}")

        print(f"  ✓ Binary mapping applied to {len(existing_binary)} features")
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
            original_dist = df_mapped[feature].value_counts().to_dict()

            df_mapped.loc[df_mapped[feature] == 0, feature] = 85
            df_mapped.loc[df_mapped[feature] == 1, feature] = 170
            df_mapped.loc[df_mapped[feature] == 2, feature] = 255

            new_dist = df_mapped[feature].value_counts().to_dict()
            print(f"    {feature}: {original_dist} -> {new_dist}")

        print(f"  ✓ Discrete mapping applied to {len(existing_discrete)} features")
    else:
        print("  No discrete features found for mapping")

    return df_mapped


def apply_rate_scaling(df, rate_features):
    """
    Scale rate features by multiplying with 255 (normalize to [0, 255] range).

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
            original_range = (df_scaled[feature].min(), df_scaled[feature].max())
            df_scaled[feature] = df_scaled[feature] * 255
            new_range = (df_scaled[feature].min(), df_scaled[feature].max())
            print(
                f"    {feature}: [{original_range[0]:.4f}, {original_range[1]:.4f}] -> [{new_range[0]:.2f}, {new_range[1]:.2f}]")

        print(f"  ✓ Rate scaling applied to {len(existing_rate)} features")
    else:
        print("  No rate features found for scaling")

    return df_scaled


def normalize_remaining_features(df, exclude_features=None):
    """
    Normalize remaining numerical features to [0, 255] range using log scaling + min-max scaling.

    Args:
        df (pd.DataFrame): Input dataframe
        exclude_features (list): Features to exclude from normalization

    Returns:
        pd.DataFrame: Dataframe with normalized features
    """
    if exclude_features is None:
        exclude_features = []

    print("Normalizing remaining numerical features to [0, 255] range...")

    df_normalized = df.copy()

    # Find numerical columns that haven't been processed yet
    processed_features = (LOG_SCALING_FEATURES + BINARY_FEATURES +
                          DISCRETE_FEATURES + RATE_FEATURES +
                          ['protocol_type', 'service', 'flag', 'classification.'] +
                          exclude_features)

    remaining_features = []
    for col in df_normalized.columns:
        if (col not in processed_features and
                df_normalized[col].dtype in ['int64', 'float64'] and
                col != 'classification.'):
            remaining_features.append(col)

    if remaining_features:
        print(f"  Features to normalize: {remaining_features}")

        for feature in remaining_features:
            original_range = (df_normalized[feature].min(), df_normalized[feature].max())

            # Apply log scaling first if the range is large
            if original_range[1] > 1000:
                df_normalized[feature] = np.log10(df_normalized[feature] + 1)

            # Apply min-max scaling to [0, 255] range
            scaler = MinMaxScaler(feature_range=(0, 255))
            df_normalized[feature] = scaler.fit_transform(df_normalized[[feature]]).flatten()

            new_range = (df_normalized[feature].min(), df_normalized[feature].max())
            print(
                f"    {feature}: [{original_range[0]:.2f}, {original_range[1]:.2f}] -> [{new_range[0]:.2f}, {new_range[1]:.2f}]")

        print(f"  ✓ Normalization applied to {len(remaining_features)} features")
    else:
        print("  No remaining features found for normalization")

    return df_normalized


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
        if (df_converted[column].dtype == 'float64' and
                column != 'classification.'):
            # Check if all values are close to integers
            if df_converted[column].notna().all():
                # Round to nearest integer and convert
                df_converted[column] = np.round(df_converted[column]).astype('int64')
                float_columns.append(column)

    if float_columns:
        print(f"  ✓ Converted {len(float_columns)} columns from float64 to int64")
    else:
        print("  No columns needed conversion")

    return df_converted


def save_preprocessed_data(df, output_dir, filename_prefix="automated_label_preprocessed"):
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

    # Save feature list for reference
    feature_file = os.path.join(output_dir, f"{filename_prefix}_feature_list.txt")
    feature_columns = [col for col in df.columns if col != 'classification.']
    with open(feature_file, 'w') as f:
        f.write("Feature List (excluding classification column):\n")
        f.write("=" * 50 + "\n")
        for i, feature in enumerate(feature_columns, 1):
            f.write(f"{i:2d}. {feature}\n")

    print(f"  ✓ Feature list saved: {feature_file}")


def display_preprocessing_summary(df_original, df_final):
    """
    Display a summary of the preprocessing transformations.

    Args:
        df_original (pd.DataFrame): Original dataframe
        df_final (pd.DataFrame): Final preprocessed dataframe
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"Original dataset shape: {df_original.shape}")
    print(f"Final dataset shape: {df_final.shape}")
    print(f"Features removed: {df_original.shape[1] - df_final.shape[1]}")

    print(f"\nData type distribution:")
    for dtype in df_final.dtypes.value_counts().index:
        count = df_final.dtypes.value_counts()[dtype]
        print(f"  {dtype}: {count} columns")

    print(f"\nValue range summary (excluding classification):")
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'classification.']

    if len(numeric_cols) > 0:
        print(f"  Minimum value: {df_final[numeric_cols].min().min():.2f}")
        print(f"  Maximum value: {df_final[numeric_cols].max().max():.2f}")
        print(
            f"  Features in [0, 255] range: {len([col for col in numeric_cols if df_final[col].min() >= 0 and df_final[col].max() <= 255])}/{len(numeric_cols)}")


def main():
    """
    Main function to run the complete automated label encoding preprocessing pipeline.
    """
    print("=" * 60)
    print("NSL-KDD Dataset Preprocessing with Automated Label Encoding")
    print("=" * 60)

    try:
        # Step 1: Load dataset
        df_original = load_dataset(CONFIG['input_file'])
        df = df_original.copy()

        # Step 2: Remove low information gain features
        df = remove_low_information_features(df, LOW_INFORMATION_FEATURES)

        # Step 3: Apply log scaling to numerical features
        df = apply_log_scaling(df, LOG_SCALING_FEATURES)

        # Step 4: Apply automated categorical encoding (replaces manual encoding)
        df = apply_automated_categorical_encoding(df)

        # Step 5: Apply binary mapping
        df = apply_binary_mapping(df, BINARY_FEATURES)

        # Step 6: Apply discrete mapping
        df = apply_discrete_mapping(df, DISCRETE_FEATURES)

        # Step 7: Apply rate scaling
        df = apply_rate_scaling(df, RATE_FEATURES)

        # Step 8: Normalize remaining features
        df = normalize_remaining_features(df)

        # Step 9: Convert to binary classification
        df = convert_to_binary_classification(df)

        # Step 10: Convert data types
        df = convert_data_types(df)

        # Step 11: Save preprocessed data
        save_preprocessed_data(df, CONFIG['output_dir'], "automated_label_preprocessed")

        # Step 12: Display summary
        display_preprocessing_summary(df_original, df)

        print("\n" + "=" * 60)
        print("✓ Automated label encoding preprocessing completed successfully!")
        print(f"✓ Final dataset shape: {df.shape}")
        print(f"✓ Output saved to: {CONFIG['output_dir']}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()