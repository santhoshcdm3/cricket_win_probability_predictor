import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
np.random.seed(42)
rows = []

def data_generation():
    for i in range(10000):
        over = np.random.randint(1, 20)
        if over <= 6:
            phase = "PP"
            run_rate = np.random.uniform(9, 13)
        elif over <= 15:
            phase = 'M'
            run_rate = np.random.uniform(10, 13)
        else:
            phase = 'Death'
            run_rate = np.random.uniform(10, 18)
        balls_remaining = (20 - over) * 6
        balls_played = (over * 6)
        wickets_remaining = np.random.randint(1, 10)
        current_score = int(over * run_rate)
        target = np.random.randint(150, 250)
        if target <= current_score:
            target = current_score + np.random.randint(10, 30)
        runs_required = target - current_score
        crr = (current_score / balls_played) * 6
        rrr = (runs_required / balls_remaining) * 6
        pitch_behavior = ['Dry','Dusty','Green','Flat','Slow']
        pitch = np.random.choice(pitch_behavior)
        if pitch in ['Flat', 'Dusty']:
            bp = 'Easy'
        elif pitch in ['Green']:
            bp = 'Medium'
        else:
            bp = 'Hard'
        batting_pitch = bp
        if pitch in ['Dry', 'Dusty']:
            sb = 3
        elif pitch in ['Slow']:
            sb = 2
        else:
            sb = 1
        spin = sb
        if pitch in ['Green']:
            pb = 3
        elif pitch in ['Slow', 'Dusty']:
            pb = 2
        else:
            pb = 1
        spin = sb
        pace = pb
        if rrr <= 6:
            ts = 'Good'
        elif rrr <= 12:
            ts = 'Average'
        else:
            ts = 'Weak'
        Team_strength = ts
        expected_recent_runs = run_rate * 12 / 6
        recent_runs = int(
            np.random.normal(
                expected_recent_runs,
                4
            )
        )
        home_awaylist = [
            'Home',
            'Away'
        ]
        home_away = np.random.choice(
            home_awaylist
        )
        if current_score >= target:
            resu = 'win'
        elif balls_remaining == 0:
            resu = 'lose'
        elif (rrr <= crr) and wickets_remaining >= 5:
            resu = 'win'
        elif (rrr <= crr + 3) and wickets_remaining >= 6:
            resu = 'win'
        elif batting_pitch == 'Easy' and wickets_remaining >= 4:
            resu = 'win'
        elif batting_pitch in ['Hard', 'Medium'] and crr >= rrr + 2:
            resu = 'lose'
        else:
            resu = np.random.choice(
                ['win', 'lose'],p=[0.4, 0.6])
        result = resu
        rows.append({
            'Over': over,
            'Balls_remaining': balls_remaining,
            'Wickets_remaining': wickets_remaining,
            'target': target,
            'Current_score': current_score,
            'Runs_required': runs_required,
            'CRR': crr,
            'RRR': rrr,
            'Pitch': pitch,
            'Batting_pitch': batting_pitch,
            'Home_Away': home_away,
            'Spin': spin,
            'Pace': pace,
            'Team_strength': Team_strength,
            'Recent_runs': recent_runs,
            'Result': result
        })
    df = pd.DataFrame(rows)
    return df


def data_clean(df):
    new_df = df.copy()
    # Dropping duplicates
    new_df = new_df.drop_duplicates()
    # Finding Null values
    print("\nNull Values:")
    print(new_df.isnull().sum())
    return new_df

def user_input():
    user_data = {
        'Over': int(input('Please enter the current over: ')),
        'Wickets_remaining': int(input('Please enter the wickets-In-Hand: ')),
        'target': int(input('Please enter the target score: ')),
        'Current_score': int(input('Please enter the current score: ')),
        'Pitch': input('Please choose pitch type(Dry/Dusty/Green/Flat/Slow): '),
        'Home_Away': input('Please choose Home/Away: ')
    }

    return user_data_preprocess(
        user_data
    )

def user_data_preprocess(user_data):
    Over_DP = user_data['Over']
    balls_remaining_DP = ((20 - Over_DP) * 6)
    wickets_remaining_DP = (user_data['Wickets_remaining'])
    target_DP = user_data['target']
    cur_score_DP = (user_data['Current_score'])
    runs_required_DP = (target_DP - cur_score_DP)
    balls_played_dp = (Over_DP * 6)
    CRR_DP = (cur_score_DP / balls_played_dp) * 6
    RRR_DP = (runs_required_DP / balls_remaining_DP) * 6
    pitch_DP = user_data['Pitch']
    if pitch_DP in ['Flat', 'Dusty']:
        batting_pitch_DP = 'Easy'
    elif pitch_DP in ['Green']:
        batting_pitch_DP = 'Medium'
    else:
        batting_pitch_DP = 'Hard'

    H_A_DP = user_data['Home_Away']
    if pitch_DP in ['Dry', 'Dusty']:
        spin_DP = 3
    elif pitch_DP in ['Slow']:
        spin_DP = 2
    else:
        spin_DP = 1

    if pitch_DP in ['Green']:
        pace_DP = 3
    elif pitch_DP in ['Slow', 'Dusty']:
        pace_DP = 2
    else:
        pace_DP = 1

    if RRR_DP <= 6:
        team_strength_DP = 'Good'
    elif RRR_DP <= 12:
        team_strength_DP = 'Average'
    else:
        team_strength_DP = 'Weak'

    expected_recent_runs_DP = (RRR_DP * 12 / 6)

    recent_runs_DP = int( np.random.normal(expected_recent_runs_DP,4))
    
    cleaned_data = {
        'Over': Over_DP,
        'Balls_remaining':balls_remaining_DP,
        'Wickets_remaining':wickets_remaining_DP,
        'target':target_DP,
        'Current_score':cur_score_DP,
        'Runs_required':runs_required_DP,
        'CRR':CRR_DP,
        'RRR':RRR_DP,
        'Pitch':pitch_DP,
        'Batting_pitch':batting_pitch_DP,
        'Home_Away':H_A_DP,
        'Spin':spin_DP,
        'Pace':pace_DP,
        'Team_strength':team_strength_DP,
        'Recent_runs':recent_runs_DP
    }

    return pd.DataFrame([cleaned_data])

def pipline(new_df):
    X = new_df.drop(columns=['Result'])
    y = new_df['Result']

    numerical_col = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_col = X.select_dtypes(include=['object']).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num',StandardScaler(),numerical_col),
            ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_col)
                    ]
    )

    model_pipeline = Pipeline(
        steps=[('preprocessor',preprocessor),(
                'classifier',
                LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )
            )
        ]
    )

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
        test_size=0.2,
        )

    model_pipeline.fit(x_train,y_train)
    print("CROSS VALIDATION")
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_accuracy = cross_val_score(
        model_pipeline,
        X,
        y,
        cv=cv,
        scoring='accuracy'
    )

    cv_precision = cross_val_score(
        model_pipeline,
        X,
        y,
        cv=cv,
        scoring='precision'
    )

    cv_recall = cross_val_score(
        model_pipeline,
        X,
        y,
        cv=cv,
        scoring='recall'
    )

    cv_f1 = cross_val_score(
        model_pipeline,
        X,
        y,
        cv=cv,
        scoring='f1'
    )

    cv_roc_auc = cross_val_score(
        model_pipeline,
        X,
        y,
        cv=cv,
        scoring='roc_auc'
    )

    print(
        f"CV Accuracy : "
        f"{cv_accuracy.mean():.4f}"
    )

    print(
        f"CV Precision: "
        f"{cv_precision.mean():.4f}"
    )

    print(
        f"CV Recall   : "
        f"{cv_recall.mean():.4f}"
    )

    print(
        f"CV F1 Score : "
        f"{cv_f1.mean():.4f}"
    )

    print(
        f"CV ROC-AUC  : "
        f"{cv_roc_auc.mean():.4f}"
    )

    print("\nCV Accuracy Scores:")
    print(cv_accuracy)

    y_pred = model_pipeline.predict(
        x_test
    )
    y_probability = model_pipeline.predict_proba(
        x_test
    )
    print("TEST SET METRICS")

    accuracy = accuracy_score(
        y_test,
        y_pred
    )
    precision = precision_score(
        y_test,
        y_pred,
        pos_label='win'
    )
    recall = recall_score(
        y_test,
        y_pred,
        pos_label='win'
    )
    f1 = f1_score(
        y_test,
        y_pred,
        pos_label='win'
    )
    win_index = list(
        model_pipeline.classes_
    ).index('win')

    roc_auc = roc_auc_score(
        y_test,
        y_probability[:, win_index]
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )
    print("CONFUSION MATRIX")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=['lose', 'win']
    )

    confusion_df = pd.DataFrame(
        cm,
        index=[
            'Actual Lose',
            'Actual Win'
        ],

        columns=[
            'Predicted Lose',
            'Predicted Win'
        ]
    )

    print(confusion_df)

    print("CLASSIFICATION REPORT")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    new_situation = user_input()

    probabilities = model_pipeline.predict_proba(new_situation)

    win_index = list(model_pipeline.classes_).index('win')

    win_percentage = (probabilities[0][win_index] * 100)

    print("MATCH PREDICTION")
    print(
        f"Predicted Win Probability: "
        f"{win_percentage:.2f}%"
    )

    return preprocessor

df = data_generation()
cleaned_df = data_clean(df)

x = pipline(cleaned_df)