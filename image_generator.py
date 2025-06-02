import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def csv_to_6x6_images(csv_input_path, img_output_path):
    """
    Convert each row of CSV file to 6x6 grayscale image

    Parameters:
    csv_input_path: Path to input CSV file
    img_output_path: Directory path to save images
    """

    # Read CSV file
    df = pd.read_csv(csv_input_path)

    # Remove class column (assuming it's the last column)
    feature_columns = df.columns[:-1]  # Exclude class column
    features_df = df[feature_columns]

    print(f"CSV file: {csv_input_path}")
    print(f"Total number of features: {len(feature_columns)}")
    print(f"Total number of data rows: {len(df)}")

    # Check if the number of features is 36
    if len(feature_columns) != 36:
        print(f"Warning: Number of features is not 36. Current: {len(feature_columns)}")
        if len(feature_columns) < 36:
            print("Padding to 36 features.")
        else:
            print("Using only the first 36 features.")

    # Create output directory
    os.makedirs(img_output_path, exist_ok=True)

    # Use original feature values without normalization
    features_array = features_df.values

    # Convert each row to image
    for idx, row in enumerate(features_array):
        # Adjust to 36 features
        if len(row) == 36:
            # Use as is if exactly 36 features
            adjusted_row = row
        elif len(row) < 36:
            # Pad with zeros if less than 36
            padding_size = 36 - len(row)
            adjusted_row = np.append(row, np.zeros(padding_size))
        else:
            # Use only first 36 if more than 36
            adjusted_row = row[:36]

        # Reshape to 6x6 array
        image_array = adjusted_row.reshape(6, 6)

        # Create and save image
        plt.figure(figsize=(2, 2))
        plt.imshow(image_array, cmap='gray', vmin=0, vmax=np.max(image_array))
        plt.axis('off')

        # Get class information
        class_label = df.iloc[idx]['class'] if 'class' in df.columns else 'unknown'

        # Generate filename
        filename = f"{img_output_path}/sample_{idx:04d}_{class_label}.png"

        # Save with DPI 600
        plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()

        if (idx + 1) % 100 == 0:
            print(f"Processed: {idx + 1}/{len(df)} images")

    print(f"All images saved successfully! Save location: {img_output_path}/")

    # Example visualization of the first image
    plt.figure(figsize=(8, 6))

    # Visualize original features
    first_row = features_array[0]
    if len(first_row) == 36:
        adjusted_first_row = first_row
    elif len(first_row) < 36:
        padding_size = 36 - len(first_row)
        adjusted_first_row = np.append(first_row, np.zeros(padding_size))
    else:
        adjusted_first_row = first_row[:36]

    image_array = adjusted_first_row.reshape(6, 6)

    plt.subplot(1, 2, 1)
    plt.imshow(image_array, cmap='gray', vmin=0, vmax=np.max(image_array))
    plt.title('6x6 Grayscale Image')
    plt.colorbar()

    # Display feature values as histogram
    plt.subplot(1, 2, 2)
    plt.hist(adjusted_first_row, bins=20, alpha=0.7)
    plt.title('Feature Values Distribution (First Sample)')
    plt.xlabel('Original Feature Values')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"{img_output_path}/example_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()

# Usage example
if __name__ == "__main__":
    # Set CSV file paths
    csv_input_path = 'your_data.csv'    # Input CSV file path
    img_output_path = 'output_images'   # Image save directory path

    # Convert CSV to images
    csv_to_6x6_images(csv_input_path, img_output_path)

    print("\n=== Conversion Complete ===")
    print("1. Each data row converted to 6x6 grayscale image")
    print("2. 36 features constitute exactly 36 pixels")
    print("3. Original feature values used (no normalization)")
    print("4. Saved as PNG format with DPI 600")
    print("5. Filename format: sample_XXXX_classname.png")
    print(f"\nUsage:")
    print(f"csv_to_6x6_images('your_data.csv', 'your_output_folder')")