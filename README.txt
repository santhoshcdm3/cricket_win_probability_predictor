CRICKET WIN PROBABILITY PREDICTION
=================================

PROJECT OVERVIEW
----------------
This project is a machine learning based Cricket Win Probability Prediction
system. It predicts the probability of a team winning a T20 cricket match
based on the current match situation.

The project uses synthetically generated cricket match-state data and a
Logistic Regression classification model. The machine learning pipeline
handles numerical and categorical features using StandardScaler and
OneHotEncoder through a ColumnTransformer.

The user can enter the current match situation through the terminal and
receive a predicted win probability.

TECHNOLOGY STACK
----------------
Python 3.x
NumPy
Pandas
Scikit-learn

Machine Learning:
- Logistic Regression
- Binary Classification
- Train/Test Split
- Stratified K-Fold Cross Validation

Preprocessing:
- StandardScaler
- OneHotEncoder
- ColumnTransformer
- Pipeline

Evaluation:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- 5-Fold Cross Validation

PROJECT WORKFLOW
----------------
1. Generate synthetic cricket match-state data.
2. Perform basic data cleaning.
3. Separate input features and target variable.
4. Detect numerical and categorical columns.
5. Apply StandardScaler to numerical features.
6. Apply OneHotEncoder to categorical features.
7. Build a Scikit-learn Pipeline.
8. Train a Logistic Regression classifier.
9. Evaluate the model using a train/test split.
10. Perform 5-fold Stratified Cross Validation.
11. Calculate classification metrics.
12. Accept a new match situation from the user.
13. Generate the predicted win probability.

FEATURES USED
-------------
Over
Balls_remaining
Wickets_remaining
target
Current_score
Runs_required
CRR
RRR
Pitch
Batting_pitch
Home_Away
Spin
Pace
Team_strength
Recent_runs

TARGET
------
Result

Target classes:
- win
- lose

FEATURE ENGINEERING
-------------------
The project derives cricket-specific match features from the current match
state.

CRR (Current Run Rate):
    Current Score / Balls Played * 6

RRR (Required Run Rate):
    Runs Required / Balls Remaining * 6

Additional derived attributes include:
- Balls remaining
- Runs required
- Batting pitch condition
- Spin condition
- Pace condition
- Team strength
- Recent runs

MACHINE LEARNING PIPELINE
-------------------------
The project uses a Scikit-learn Pipeline so preprocessing and model
training are handled together.

Numerical features:
    StandardScaler

Categorical features:
    OneHotEncoder

Model:
    LogisticRegression

This approach keeps the preprocessing and classifier connected and helps
maintain the same transformation process during training, evaluation and
prediction.

CROSS VALIDATION
----------------
The project uses StratifiedKFold with 5 folds.

Cross-validation metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Stratification maintains the class distribution across the folds.

MODEL EVALUATION
----------------
The model is evaluated on a separate test set using:

Accuracy:
Measures the overall percentage of correct predictions.

Precision:
Measures how many predicted wins were actually wins.

Recall:
Measures how many actual wins were correctly identified.

F1 Score:
Balances precision and recall.

ROC-AUC:
Measures the model's ability to distinguish between win and lose classes
using prediction probabilities.

Confusion Matrix:
Shows actual versus predicted win/lose classifications.

Classification Report:
Provides precision, recall, F1 score and support for each class.

USER PREDICTION
---------------
The application accepts:

- Current over
- Wickets in hand
- Target score
- Current score
- Pitch type
- Home/Away status

The system derives the remaining match features and sends the resulting
data through the same trained preprocessing and Logistic Regression
pipeline.

The final output is:

    Predicted Win Probability: XX.XX%

PROJECT STRUCTURE
-----------------
cricket_win_prediction/
|
|-- cricket_prediction.py
|-- README.txt
|-- ABOUT_PROJECT.txt
|
`-- requirements.txt

REQUIREMENTS
------------
Install the required libraries:

    pip install numpy pandas scikit-learn

RUN PROJECT
-----------
Run the Python file:

    python cricket_prediction.py

The terminal will ask for the current match situation.

Example:

    Please enter the current over: 12
    Please enter the wickets-In-Hand: 6
    Please enter the target score: 190
    Please enter the current score: 110
    Please choose pitch type(Dry/Dusty/Green/Flat/Slow): Flat
    Please choose Home/Away: Home

The model then displays evaluation metrics and the predicted win
probability.

IMPORTANT PROJECT NOTE
----------------------
The training dataset is synthetically generated. The Result target is
created using predefined cricket match-state rules.

Therefore, this project demonstrates an end-to-end machine learning
workflow, feature engineering, preprocessing, classification, cross
validation and probability prediction.

The model's evaluation scores should not be interpreted as real-world
cricket prediction accuracy.

FUTURE IMPROVEMENTS
-------------------
Possible future improvements include:

- Replace synthetic data with real T20/IPL ball-by-ball data.
- Add team-specific historical performance.
- Add player-level statistics.
- Add venue information.
- Add innings and batting/bowling strength.
- Compare Logistic Regression with Random Forest, XGBoost and other
  classification algorithms.
- Add calibration analysis for probability predictions.
- Build a Streamlit web interface.
- Store historical predictions and match results.
- Add visual dashboards for model performance.
