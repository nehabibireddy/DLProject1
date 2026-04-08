import os
import io
import base64

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

MODEL_OPTIONS = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}


def _train():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    trained = {}
    for name, clf in MODEL_OPTIONS.items():
        clf.fit(X_train, y_train)
        trained[name] = clf
    return trained, X_test, y_test, iris.target_names, iris.feature_names


models, X_test, y_test, target_names, feature_names = _train()


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_names=list(MODEL_OPTIONS.keys()),
        feature_names=list(feature_names),
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    model_name = data.get("model", "Random Forest")
    features = [
        float(data["sepal_length"]),
        float(data["sepal_width"]),
        float(data["petal_length"]),
        float(data["petal_width"]),
    ]

    clf = models[model_name]
    input_df = pd.DataFrame([features], columns=feature_names)
    pred_idx = clf.predict(input_df)[0]
    probs = clf.predict_proba(input_df)[0].tolist()
    predicted_class = str(target_names[pred_idx]).capitalize()

    # Test-set accuracy + confusion matrix
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    cm_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return jsonify(
        {
            "prediction": predicted_class,
            "probabilities": {
                str(target_names[i]).capitalize(): round(probs[i], 4)
                for i in range(len(target_names))
            },
            "accuracy": f"{acc * 100:.2f}%",
            "confusion_matrix": cm_b64,
        }
    )


@app.route("/compare")
def compare():
    rows = []
    for name, clf in models.items():
        preds = clf.predict(X_test)
        rows.append(
            {"model": name, "accuracy": f"{accuracy_score(y_test, preds) * 100:.2f}%"}
        )
    return jsonify(rows)


if __name__ == "__main__":
    app.run(debug=True)
