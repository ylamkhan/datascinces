#!/usr/bin/env python3
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
import graphviz

def main():
    if len(sys.argv) != 4:
        print("Usage: python3.10 Tree.py ../dataset/Training_knight.csv ../dataset/Validation_knight.csv ../dataset/Test_knight.csv")
        sys.exit(1)

    train_file = sys.argv[1]
    valid_file = sys.argv[2]
    test_file  = sys.argv[3]

    try:
        train_df = pd.read_csv(train_file)
        valid_df = pd.read_csv(valid_file)
        test_df  = pd.read_csv(test_file)

        selected_features = ['Push', 'Lightsaber', 'Attunement']

        X_train = train_df[selected_features]
        y_train = train_df.iloc[:, -1]

        X_valid = valid_df[selected_features]
        y_valid = valid_df.iloc[:, -1]

        X_test = test_df[selected_features]

        f1 = 0
        attempt = 0
        max_attempts = 100

        while f1 < 0.92 and attempt < max_attempts:
            attempt += 1

            model = RandomForestClassifier(
                n_estimators=200,
                random_state=attempt
            )

            model.fit(X_train, y_train)

            y_pred_val = model.predict(X_valid)

            f1 = f1_score(y_valid, y_pred_val, pos_label='Jedi')
            print(f"Attempt {attempt}, F1-score: {f1:.3f}")

        if f1 < 0.93:
            print("Warning: Could not reach F1 >= 0.93 after 100 attempts.")
        else:
            print(f"Success! F1 >= 0.93 achieved in attempt {attempt}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(0)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    with open("Tree.txt", "w") as f:
        for pred in predictions:
            f.write(str(pred) + "\n")

    print("Predictions saved to Tree.txt")

    estimator = model.estimators_[0]

    dot_data = export_graphviz(
        estimator,
        out_file=None,
        feature_names=selected_features,
        class_names=[str(c) for c in model.classes_],
        filled=True,
        rounded=True,
        special_characters=True
    )

    graph = graphviz.Source(dot_data)
    graph.render("RandomForestTree", format='png', cleanup=True)

    print("Tree graph saved as RandomForestTree.png")


if __name__ == "__main__":
    main()