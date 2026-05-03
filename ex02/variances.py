import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def __calcule_variances(df):
    try:
        skills = df.select_dtypes(include=['float64'])
        scaler = StandardScaler()
        skills_scaled = scaler.fit_transform(skills)
        pca = PCA()
        pca.fit(skills_scaled)
        variances = pca.explained_variance_ratio_ * 100
        return variances
    except Exception as e:
        print(f"Error in __calcule_variances: {e}")


def __calcule_cumulative_variances(__va):
    try:
        cumulative_variances = np.cumsum(__va)
        return cumulative_variances
    except Exception as e:
        print(f"Error in __calcule_cumulative_variances: {e}")


def __display(df, __va, __cva):
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(
            range(1, len(__cva)+1),
            __cva, marker='o', color='blue', label='Cumulative Variance')
        n_components_90 = np.argmax(__cva >= 90) + 1
        plt.axhline(y=90, color='r', linestyle='--', label='90% variance')
        plt.axvline(x=n_components_90,
                    color='g', linestyle='--',
                    label=f'{n_components_90} components')
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance (%)")
        plt.title("Cumulative Variance Explained by Components")
        plt.xticks(range(1, len(__cva)+1))
        plt.legend()
        plt.grid(True)
        plt.savefig('variance.png')
        # plt.show()
        plt.close()
    except Exception as e:
        print(f"Error in __display: {e}")


def main():
    try:
        df = pd.read_csv("../dataset/Training_knight.csv")
        print("Variances (Percentage):")
        __va = __calcule_variances(df)
        print(__va)
        print("Cumulative Variances (Percentage):")
        __cva = __calcule_cumulative_variances(__va)
        print(__cva)
        __display(df, __va, __cva)
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
