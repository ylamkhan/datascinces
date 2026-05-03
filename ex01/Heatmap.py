import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def create_full_heatmap(data_file):
    try:
        df = pd.read_csv(data_file)
        if 'knight' in df.columns:
            df['knight'] = df['knight'].map({'Jedi': 1, 'Sith': 0})
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        plt.figure(figsize=(20, 16))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot_kws={'size': 8}
        )

        plt.title('Correlation Heatmap of All Features', fontsize=18, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('full_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(0)



def main():
    if len(sys.argv) < 2:
        print("Usage: python3.10 Heatmap.py ../dataset/Training_knight.csv")
        sys.exit(1)
    data_file = sys.argv[1]
    print("Generating full correlation heatmap...")
    corr_matrix = create_full_heatmap(data_file)

if __name__ == "__main__":
    main()
