Heart Disease Prediction
A small Python project that predicts the presence of heart disease from patient clinical measurements. It implements a binary classification Perceptron completely from scratch with NumPy — no scikit-learn. The program reports custom evaluation metrics (accuracy, precision, recall, F1-score, and a confusion matrix), plots training errors and predictions, and includes a Matplotlib interface for entering a single patient's values and requesting a real-time prediction.
Dataset
The data comes from Kaggle's Heart Disease Dataset. It combines four databases (Cleveland, Hungary, Switzerland, and Long Beach V) and contains 1,025 patient records with 14 columns.
The program expects a CSV with these columns:
RoleDataset columnInput featurecp (chest pain type)Input featurethalach (maximum heart rate achieved)Input featureca (major vessels colored by fluoroscopy)Input featureoldpeak (ST depression induced by exercise)Input featureexang (exercise-induced angina)Input featurethal (thalassemia type)Prediction targettarget (1 = disease, 0 = healthy)
Download the dataset from Kaggle and place its CSV at data/heart.csv. The filename must match exactly because main.py reads that path. Only the six features listed above are used; the remaining columns (age, sex, trestbps, chol, fbs, restecg, slope) are ignored to reduce noise and improve generalization on this simple linear model.
Models and workflow
The selected columns are converted to numeric NumPy arrays by PrepareData in src/preprocessing.py.
The rows are shuffled using seed 42 and split into 80% training and 20% test data.
Input features are standardized with z-scores using the training set's mean and standard deviation; those same values are applied to the test set and to interactive user input.
A Perceptron is trained with Rosenblatt's learning rule, a learning rate of 0.01, and 100 epochs:
A bias term is added to every sample through the bias trick (_add_bias).
A step (threshold) activation function maps the weighted sum to a 0/1 class label.
Weights are updated online after each misclassified sample: $w \leftarrow w + \eta,(y_i - \hat{y}i),x{b,i}$.
The number of misclassifications per epoch is recorded for plotting.
Performance is evaluated with metrics implemented from scratch — accuracy, precision, recall, F1-score, and a confusion matrix — on both the training and test sets.
With the six selected features the model reaches roughly 77.6% test accuracy and 84.5% recall, which is preferable to the full 13-feature setup (about 75.1% accuracy) because recall matters most in a medical screening context. A Perceptron is a linear classifier, so these figures are not guaranteed to generalize to other datasets or splits.
Requirements
Python 3.10 or newer
Packages listed in requirements.txt:
numpy
pandas
matplotlib

No machine learning framework such as scikit-learn is used; the Perceptron and all evaluation metrics are implemented from scratch with NumPy.
Installation and usage
Clone or download this repository, ensure the dataset is at data/heart.csv, then run the following from the project directory:
python -m venv .venv

Activate the environment:
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

Install dependencies and start the program:
python -m pip install -r requirements.txt
python main.py

The program prints the model's bias and weights, then the training and test metrics with the confusion matrix. It opens Matplotlib windows showing the training error history, the test-set confusion matrix, and a comparison of actual versus predicted classes. Finally, a prediction window appears where you enter values for cp, thalach, ca, oldpeak, exang, and thal, then click Predict to see whether the patient is classified as at risk (1) or healthy (0).
Project structure
.
├── main.py                  # Training, evaluation metrics, plots, and prediction UI
├── requirements.txt         # Python dependencies
├── data/
│   └── heart.csv            # Kaggle dataset (place it here)
└── src/
    ├── preprocessing.py     # Feature selection, standardization, train/test split
    └── classification.py    # Perceptron model implemented from scratch

Data attribution
Please follow Kaggle's terms and the dataset creator's licensing and attribution requirements when using or redistributing the data. This project does not include a separate dataset license.