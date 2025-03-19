import json
import random
import os

def split_json(input_file, output_dir, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, seed=42):
    """
    Splits a JSON dataset into train, test, and validation sets.

    :param input_file: Path to the input JSON file.
    :param output_dir: Directory where split files will be saved.
    :param train_ratio: Proportion of data to be used for training.
    :param test_ratio: Proportion of data to be used for testing.
    :param val_ratio: Proportion of data to be used for validation.
    :param seed: Random seed for reproducibility.
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Define output file names
    train_file = 'train.json'
    test_file = 'test.json'
    val_file = 'val.json'

    # Read JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure the data is a list
    if not isinstance(data, list):
        raise ValueError("JSON data should be a list of records.")

    # Shuffle data
    random.shuffle(data)

    # Calculate split indices
    total = len(data)
    train_end = int(train_ratio * total)
    test_end = train_end + int(test_ratio * total)

    # Split data
    train_data = data[:train_end]
    test_data = data[train_end:test_end]
    val_data = data[test_end:]

    # Handle any remaining data by adding to the training set
    remaining = total - (len(train_data) + len(test_data) + len(val_data))
    if remaining > 0:
        train_data += data[-remaining:]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save split data
    with open(os.path.join(output_dir, train_file), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, test_file), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, val_file), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)

    # Print summary
    print(f"Data successfully split and saved to '{output_dir}' directory.")
    print(f"Training set size: {len(train_data)}")
    print(f"Testing set size: {len(test_data)}")
    print(f"Validation set size: {len(val_data)}")

if __name__ == "__main__":
    # Define file paths
    input_file = 'preprocessed/energy_m-3m.json'       # Input JSON file
    output_dir = 'preprocessed/splits'          # Output directory for splits

    # Call the split function
    split_json(input_file, output_dir)
