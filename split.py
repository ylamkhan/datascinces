import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    if len(sys.argv) != 2:
        print("Usage: python split.py Train_knight.csv")
        return

    input_file = sys.argv[1]

    # Load CSV
    df = pd.read_csv(input_file)

    # Split data (80% train, 20% validation)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,   # 20% validation
        random_state=42, # ensures reproducibility
        shuffle=True
    )

    # Save files
    train_df.to_csv("dataset/Training_knight.csv", index=False)
    val_df.to_csv("dataset/Validation_knight.csv", index=False)

    print("Split completed:")
    print(f"Training: {len(train_df)} rows")
    print(f"Validation: {len(val_df)} rows")

if __name__ == "__main__":
    main()