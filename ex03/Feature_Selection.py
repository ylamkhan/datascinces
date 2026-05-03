import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(df):
    """
    Calculate VIF for each numeric feature
    """
    features = df.select_dtypes(include=['float64'])
    vif_data = pd.DataFrame()
    vif_data['Feature'] = features.columns
    vif_data['VIF'] = [
        variance_inflation_factor(features.values, i)
        for i in range(features.shape[1])
        ]
    vif_data['Tolerance'] = 1 / vif_data['VIF']
    return vif_data


def select_features(df, threshold=5.0):
    """
    Iteratively remove features with VIF > threshold
    """
    features = df.select_dtypes(include=['float64'])
    while True:
        vif = pd.DataFrame()
        vif['Feature'] = features.columns
        vif['VIF'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]

        max_vif = vif['VIF'].max()
        if max_vif <= threshold:
            break
        remove_feature = vif.sort_values('VIF', ascending=False)['Feature'].iloc[0]
        print(f"Removing '{remove_feature}' with VIF={max_vif:.2f}")
        features = features.drop(columns=[remove_feature])

    return features


def main():

    df = pd.read_csv("../dataset/Training_knight.csv")

    print("=== VIF of all features ===")
    vif_data = calculate_vif(df)
    print(vif_data)

    print("\n=== Selecting features with VIF <= 5 ===")
    selected_features = select_features(df, threshold=5.0)
    print("\nSelected Features:")
    print(selected_features.columns.tolist())


if __name__ == "__main__":
    main()
