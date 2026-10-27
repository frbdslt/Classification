# Heart Disease Prediction

A small educational Python project that implements a binary Perceptron with NumPy, without a machine learning framework. It prints accuracy, precision, recall, F1-score, and a confusion matrix, plots training results, and provides a Matplotlib interface for trying one sample.

This project is for learning and experimentation. Its output is not a medical diagnosis and should not be used to make health decisions.

## Dataset

The project reads `data/heart.csv`, which is included in this repository. It expects these columns:

| Role | Column |
| --- | --- |
| Input feature | `cp` |
| Input feature | `thalach` |
| Input feature | `ca` |
| Input feature | `oldpeak` |
| Input feature | `exang` |
| Input feature | `thal` |
| Prediction target | `target` (0 or 1) |

Other columns are ignored. Before splitting, the program removes repeated rows based on the six selected features and the target. This helps prevent identical examples from appearing in both the training and test sets. In the included file, that leaves 293 unique feature-and-target combinations from 1,025 rows. The split is shuffled with seed 42 and uses 80% for training and 20% for testing.

The features are standardized using the training set's mean and standard deviation. The same values are used to standardize test examples and values entered in the interface. The Perceptron uses a learning rate of 0.01 and runs for up to 100 epochs.

The dataset contains repeated examples, and this small project uses a single random split. Its scores should be treated as an example of how the code works, not as evidence of real-world medical performance. Scores can differ from older results because duplicate examples are now removed before splitting.

## Requirements

- Python 3.10 or newer
- Packages in `requirements.txt`: NumPy, pandas, and Matplotlib

## Installation and usage

From the project directory, create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with `source .venv/bin/activate`.

Install dependencies and run the program:

```bash
python -m pip install -r requirements.txt
python main.py
```

The program prints training and test metrics, then opens plots and the single-sample interface. The interface checks that entered values are numeric and within simple allowed ranges. It displays a model class output, which is not a diagnosis.

## Project structure

```text
main.py                  Training, evaluation, plots, and prediction interface
requirements.txt         Python dependencies
data/heart.csv           Dataset
src/preprocessing.py     Feature selection, validation, standardization, and split
src/classification.py    Perceptron implemented from scratch
```

Please follow the dataset creator's license and attribution requirements when using or redistributing the data. This project does not include a separate dataset license.
