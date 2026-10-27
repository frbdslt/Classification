import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

from src.preprocessing import PrepareData, StandardizeFeatures, ApplyStandardization, TrainTestSplit
from src.classification import Perceptron


def AccuracyScore(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return np.mean(y_true == y_pred)


def ConfusionMatrix(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([[tn, fp], [fn, tp]])


def PrecisionScore(y_true, y_pred):
    cm = ConfusionMatrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    denominator = tp + fp
    if denominator == 0:
        return 0.0
    return tp / denominator


def RecallScore(y_true, y_pred):
    cm = ConfusionMatrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    denominator = tp + fn
    if denominator == 0:
        return 0.0
    return tp / denominator


def F1Score(y_true, y_pred):
    precision = PrecisionScore(y_true, y_pred)
    recall = RecallScore(y_true, y_pred)
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2 * precision * recall / denominator


def PlotErrorHistory(errors_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(errors_history) + 1), errors_history, marker="o")
    plt.title("Perceptron Training Errors per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Misclassifications")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def PlotConfusionMatrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()

    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=18)

    plt.tight_layout()
    plt.show()


def PlotPredictionComparison(y_true, y_pred, num_samples=50):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    count = min(num_samples, len(y_true))
    indices = np.arange(count)

    plt.figure(figsize=(12, 6))
    plt.plot(indices, y_true[:count], marker="o", label="Actual")
    plt.plot(indices, y_pred[:count], marker="s", label="Predicted")
    plt.title(f"Actual vs Predicted Classes (First {count} Test Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Class")
    plt.yticks([0, 1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def PrintResults(name, y_true, y_pred):
    accuracy = AccuracyScore(y_true, y_pred)
    precision = PrecisionScore(y_true, y_pred)
    recall = RecallScore(y_true, y_pred)
    f1 = F1Score(y_true, y_pred)
    cm = ConfusionMatrix(y_true, y_pred)

    print(f"\n--- {name} Results ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nMatrix format:")
    print("[[TN, FP]")
    print(" [FN, TP]]")

    return cm


def LaunchPredictionUI(model, feature_names, train_mean, train_std):
    fig = plt.figure(figsize=(8, 10))
    fig.suptitle("Model Prediction for One Sample", fontsize=16)

    text_boxes = []
    default_values = {
        "cp": "0",
        "thalach": "150",
        "exang": "0",
        "oldpeak": "1.0",
        "ca": "0",
        "thal": "2",
    }

    start_y = 0.88
    step_y = 0.055

    for i, feature in enumerate(feature_names):
        y_pos = start_y - i * step_y
        ax_box = fig.add_axes([0.30, y_pos, 0.40, 0.035])
        text_box = TextBox(ax_box, f"{feature}: ", initial=default_values.get(feature, "0"))
        text_boxes.append(text_box)

    ax_button = fig.add_axes([0.35, 0.08, 0.30, 0.06])
    predict_button = Button(ax_button, "Predict", color="lightblue", hovercolor="skyblue")

    result_ax = fig.add_axes([0.15, 0.01, 0.70, 0.05])
    result_ax.axis("off")
    result_text = result_ax.text(
        0.5,
        0.5,
        "Educational model output, not a medical diagnosis",
        ha="center",
        va="center",
        fontsize=12,
        color="darkblue",
    )

    def on_predict(event):
        try:
            user_values = []
            for text_box in text_boxes:
                value = float(text_box.text.strip())
                user_values.append(value)

            cp, thalach, ca, oldpeak, exang, thal = user_values
            if cp not in (0, 1, 2, 3):
                raise ValueError("cp must be 0, 1, 2, or 3")
            if thalach <= 0:
                raise ValueError("thalach must be greater than 0")
            if ca not in (0, 1, 2, 3, 4):
                raise ValueError("ca must be 0, 1, 2, 3, or 4")
            if oldpeak < 0:
                raise ValueError("oldpeak cannot be negative")
            if exang not in (0, 1):
                raise ValueError("exang must be 0 or 1")
            if thal not in (0, 1, 2, 3):
                raise ValueError("thal must be 0, 1, 2, or 3")

            x_user = np.array(user_values, dtype=np.float64).reshape(1, -1)
            x_user_norm = ApplyStandardization(x_user, train_mean, train_std)
            prediction = model.predict(x_user_norm)[0, 0]

            if prediction == 1:
                result_text.set_text("Model output: class 1 (risk indicator)")
                result_text.set_color("red")
            else:
                result_text.set_text("Model output: class 0 (no risk indicator)")
                result_text.set_color("green")

            fig.canvas.draw_idle()

        except ValueError as error:
            result_text.set_text(f"Invalid input: {error}")
            result_text.set_color("orange")
            fig.canvas.draw_idle()

    predict_button.on_clicked(on_predict)

    plt.show()


def main():
    print("===== Perceptron Binary Classification =====")

    df = pd.read_csv("data/heart.csv")

    X, y, feature_names = PrepareData(df)

    X_train, X_test, y_train, y_test = TrainTestSplit(X, y, test_ratio=0.2, seed=42)

    X_train_norm, train_mean, train_std = StandardizeFeatures(X_train)
    X_test_norm = ApplyStandardization(X_test, train_mean, train_std)

    model = Perceptron(learning_rate=0.01, epochs=100)
    model.fit(X_train_norm, y_train)

    y_train_pred = model.predict(X_train_norm)
    y_test_pred = model.predict(X_test_norm)

    print("\nBias:")
    print(f"{model.w[0, 0]:.4f}")

    print("\nWeights:")
    for i, name in enumerate(feature_names):
        print(f"{name:<10}: {model.w[i + 1, 0]:.4f}")

    train_cm = PrintResults("Training", y_train, y_train_pred)
    test_cm = PrintResults("Test", y_test, y_test_pred)

    PlotErrorHistory(model.errors_history)
    PlotConfusionMatrix(test_cm, title="Confusion Matrix - Test Set")
    PlotPredictionComparison(y_test, y_test_pred, num_samples=50)

    LaunchPredictionUI(model, feature_names, train_mean, train_std)


if __name__ == "__main__":
    main()
