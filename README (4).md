# Women in AI Fellowship Admission Prediction Model

A supervised machine learning project that predicts whether an applicant will be **admitted to a fellowship**, built as a binary classification problem with Python and scikit-learn.

## Project overview

The notebook walks through a complete beginner-friendly ML workflow: defining the problem, exploring the data, preparing features, splitting the data, training two classifiers, and comparing their accuracy.

## Dataset

- **File:** `data/fellowship_data.csv`
- **Size:** 100 rows, 6 columns, no missing values
- **Columns:** `Name`, `Gender`, `Age`, `City`, `Graduate`, `Accepted`
- **Target:** `Accepted` (60 Yes, 40 No)
- **Features used:** `Gender`, `Age`, `City`, `Graduate`

Applicants come from 11 Nigerian cities, with Ilorin the most common (62 of 100).

## Workflow

1. **Problem definition:** predict fellowship admission (Yes or No).
2. **Data exploration:** `head()`, `info()`, `describe()`, missing-value check, and a bar chart of the acceptance distribution.
3. **Feature preparation:** one-hot encoding of the categorical columns with `pd.get_dummies(..., drop_first=True)`, and conversion of the target to 0 and 1.
4. **Train/test split:** 80% training (80 rows) and 20% testing (20 rows), using `random_state=42` and `stratify=y` to keep the class balance.
5. **Feature scaling:** `StandardScaler` fitted on the training set only.
6. **Modelling:**
   - Logistic Regression (on scaled features)
   - Decision Tree Classifier (on unscaled features)
7. **Evaluation:** accuracy, confusion matrix and classification report, plus a bar chart comparing the models.

## Results

| Model | Test accuracy |
| --- | --- |
| Logistic Regression | 100% |
| Decision Tree | 100% |

Logistic Regression's confusion matrix shows all 20 test applicants classified correctly.

### Limitations

Perfect accuracy on a 20-row test set should not be read as a sign of a perfect model. The dataset is very small, and one-hot encoding produced 38 features for only 80 training rows, which makes it easy for a model to fit the data without learning patterns that generalise. These results are best treated as a learning exercise, and the next step would be to test on more data and use cross-validation.

## Tech stack

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## How to run

1. Clone the repository:
   ```bash
   git clone https://github.com/carinaigbodor/<repo-name>.git
   cd <repo-name>
   ```
2. Install the dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
   ```
3. Make sure `fellowship_data.csv` is inside a `data/` folder.
4. Open the notebook and run all cells:
   ```bash
   jupyter notebook Women_in_AI_Prediction_model.ipynb
   ```

## Possible improvements

- Collect more data and test on a larger, more varied sample.
- Use cross-validation and tune hyperparameters with `GridSearchCV`.
- Try other models, such as Random Forest.
- Treat `Age` as a numeric feature instead of one-hot encoding it.
- Add a short section on fairness, since admission predictions affect real people.

## Author

**Carina Uregbo Igbodor**
AI/ML Engineering Fellow, Women in AI Fellowship
[LinkedIn](https://linkedin.com/in/carina-igbodor-753690195) | carinaigbodor0@gmail.com
