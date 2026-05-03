import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

def main():
    if len(sys.argv) != 4:
        print("Usage: python3.10 democracy.py ../dataset/Training_knight.csv ../dataset/Validation_knight.csv ../dataset/Test_knight.csv")
        return

    try:
        train = pd.read_csv(sys.argv[1])
        val = pd.read_csv(sys.argv[2])
        test = pd.read_csv(sys.argv[3])

        X_train, y_train = train.drop(columns=['knight']), train['knight']
        X_val, y_val = val.drop(columns=['knight']), val['knight']
        X_test = test
        pipe_lr = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=2000))])
        pipe_dt = Pipeline([('s', StandardScaler()), ('dt', DecisionTreeClassifier(max_depth=15))])
        pipe_knn = Pipeline([('s', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=3))])
        voting_clf = VotingClassifier(
            estimators=[('Logistic Regression', pipe_lr), 
                        ('Decision Tree', pipe_dt), 
                        ('KNN', pipe_knn)],
            voting='hard'
        )
        model_names = ['Logistic Regression', 'Decision Tree', 'KNN', 'Voting Classifier']
        models = [pipe_lr, pipe_dt, pipe_knn, voting_clf]
        scores = []

        for name, model in zip(model_names, models):
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = f1_score(y_val, preds, average='weighted')
            scores.append(score)
            print(f"{name} F1-Score: {score:.4f}")

        
        test_preds = voting_clf.predict(X_test)
        with open("Voting.txt", "w") as f:
            for pred in test_preds:
                f.write(f"{pred}\n")

        plt.figure(figsize=(10, 6))
        sns.barplot(x=model_names, y=scores, hue=model_names, palette='viridis', legend=False)
        plt.axhline(0.94, color='red', linestyle='--', label='Required Threshold (94%)')
        plt.ylim(0.9, 1.0)
        plt.ylabel('F1-Score')
        plt.title('Model Performance Comparison - Exercise 06')
        plt.legend(loc='lower right')
        plt.savefig('model_comparison.png')
        print("Comparison graph saved as 'model_comparison.png'")
        plt.close()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(0)

if __name__ == "__main__":
    main()