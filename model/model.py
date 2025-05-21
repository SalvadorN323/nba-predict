# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.utils.class_weight import compute_class_weight
# import joblib
# import numpy as np  # Import numpy

# # Load the data (assuming CSV file is named 'nba_games.csv')
# df = pd.read_csv('regular_season_totals_2010_2024.csv')

# # Feature Engineering:
# # 1. Calculate rolling averages (e.g., for the last 5 games)
# df['PTS_ROLLING_AVG'] = df['PTS'].rolling(window=5).mean().shift(-5)  # Moving average for points
# df['REB_ROLLING_AVG'] = df['REB'].rolling(window=5).mean().shift(-5)  # Moving average for rebounds
# df['AST_ROLLING_AVG'] = df['AST'].rolling(window=5).mean().shift(-5)  # Moving average for assists

# # 2. Calculate difference between home and away stats
# df['PTS_DIFF'] = df['PTS'] - df['PTS'].shift(-1)
# df['FG_PCT_DIFF'] = df['FG_PCT'] - df['FG_PCT'].shift(-1)
# df['REB_DIFF'] = df['REB'] - df['REB'].shift(-1)
# df['AST_DIFF'] = df['AST'] - df['AST'].shift(-1)
# df['STL_DIFF'] = df['STL'] - df['STL'].shift(-1)
# df['BLK_DIFF'] = df['BLK'] - df['BLK'].shift(-1)
# df['FT_PCT_DIFF'] = df['FT_PCT'] - df['FT_PCT'].shift(-1)

# # Fill missing values with 0
# df.fillna(0, inplace=True)

# # Encode the target variable (Win/Loss)
# df['WL'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)

# # Selecting the features
# features = ['PTS_DIFF', 'FG_PCT_DIFF', 'REB_DIFF', 'AST_DIFF', 'STL_DIFF', 'BLK_DIFF', 'FT_PCT_DIFF', 
#             'PTS_ROLLING_AVG', 'REB_ROLLING_AVG', 'AST_ROLLING_AVG']
# X = df[features]
# y = df['WL']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Handling class imbalance using class weights
# class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)  # Fix: Use numpy array
# class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# # Initialize the base models
# model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
# model_gb = GradientBoostingClassifier(random_state=42)
# model_xgb = XGBClassifier(eval_metric='logloss', random_state=42)

# # Create the Voting Classifier (using soft voting to predict the probabilities)
# voting_clf = VotingClassifier(estimators=[('rf', model_rf), ('gb', model_gb), ('xgb', model_xgb)], voting='soft')

# # Train the Voting Classifier
# voting_clf.fit(X_train, y_train)

# # Evaluate the model's performance on the test set
# y_pred = voting_clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Voting Classifier Accuracy: {accuracy:.4f}")

# # Calculate Win/Loss Percentages using predict_proba
# # predict_proba gives the probabilities for each class (0 for Loss, 1 for Win)
# probabilities = voting_clf.predict_proba(X_test)

# # Probability of win for each game (column 1 is the probability of class '1' - Win)
# win_probabilities = probabilities[:, 1]

# # Print win/loss probabilities (as percentages)
# for i in range(len(win_probabilities)):
#     print(f"Game {i+1}: Team A Win Probability: {win_probabilities[i]*100:.2f}%")

# # Hyperparameter tuning with GridSearchCV (only for RandomForestClassifier for demonstration)
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2']
# }

# grid_search = GridSearchCV(estimator=model_rf, 
#                            param_grid=param_grid, 
#                            cv=3, 
#                            n_jobs=-1, 
#                            verbose=2)
# grid_search.fit(X_train, y_train)

# # Print the best parameters and best score from the grid search
# print(f"Best parameters (RandomForest): {grid_search.best_params_}")
# print(f"Best cross-validation score (RandomForest): {grid_search.best_score_:.4f}")

# # Save the best model from GridSearchCV (RandomForest in this case)
# best_rf_model = grid_search.best_estimator_
# joblib.dump(best_rf_model, 'nba_game_predictor_rf_model_best.joblib')

# # Cross-validation to check model performance
# cv_scores = cross_val_score(voting_clf, X, y, cv=5, scoring='accuracy')
# print(f"Cross-validation scores: {cv_scores}")
# print(f"Mean cross-validation score: {cv_scores.mean():.4f}")


# MAIN MODEL CODE

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
import joblib
import numpy as np  # Import numpy

# Load the data (assuming CSV file is named 'nba_games.csv')
df = pd.read_csv('regular_season_totals_2010_2024.csv')

# Feature Engineering:
# 1. Calculate rolling averages (e.g., for the last 5 games)
df['PTS_ROLLING_AVG'] = df['PTS'].rolling(window=5).mean().shift(-5)  # Moving average for points
df['REB_ROLLING_AVG'] = df['REB'].rolling(window=5).mean().shift(-5)  # Moving average for rebounds
df['AST_ROLLING_AVG'] = df['AST'].rolling(window=5).mean().shift(-5)  # Moving average for assists

# 2. Calculate difference between home and away stats
df['PTS_DIFF'] = df['PTS'] - df['PTS'].shift(-1)
df['FG_PCT_DIFF'] = df['FG_PCT'] - df['FG_PCT'].shift(-1)
df['REB_DIFF'] = df['REB'] - df['REB'].shift(-1)
df['AST_DIFF'] = df['AST'] - df['AST'].shift(-1)
df['STL_DIFF'] = df['STL'] - df['STL'].shift(-1)
df['BLK_DIFF'] = df['BLK'] - df['BLK'].shift(-1)
df['FT_PCT_DIFF'] = df['FT_PCT'] - df['FT_PCT'].shift(-1)

# Fill missing values with 0
df.fillna(0, inplace=True)

# Encode the target variable (Win/Loss)
df['WL'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)

# Selecting the features
features = ['PTS_DIFF', 'FG_PCT_DIFF', 'REB_DIFF', 'AST_DIFF', 'STL_DIFF', 'BLK_DIFF', 'FT_PCT_DIFF', 
            'PTS_ROLLING_AVG', 'REB_ROLLING_AVG', 'AST_ROLLING_AVG']
X = df[features]
y = df['WL']

# Target variables for score prediction (scores of both teams)
y_team_a_score = df['PTS']  # Scores for Team A
y_team_b_score = df['PTS'].shift(-1).fillna(0)  # Scores for Team B, shifted one step back to match the pair

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, y_train_team_a, y_test_team_a, y_train_team_b, y_test_team_b = train_test_split(
    X, y, y_team_a_score, y_team_b_score, test_size=0.2, random_state=42)

# Handling class imbalance using class weights for win/loss
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)  # Fix: Use numpy array
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Initialize the base models for classification and regression
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
model_gb = GradientBoostingClassifier(random_state=42)
model_xgb = XGBClassifier(eval_metric='logloss', random_state=42)

# Initialize models for score prediction (regression)
reg_rf_team_a = RandomForestRegressor(n_estimators=100, random_state=42)
reg_rf_team_b = RandomForestRegressor(n_estimators=100, random_state=42)

# Create the Voting Classifier (using soft voting to predict the probabilities)
voting_clf = VotingClassifier(estimators=[('rf', model_rf), ('gb', model_gb), ('xgb', model_xgb)], voting='soft')

# Train the classifiers and regressors
voting_clf.fit(X_train, y_train)
reg_rf_team_a.fit(X_train, y_train_team_a)
reg_rf_team_b.fit(X_train, y_train_team_b)
 
# Evaluate the model's performance on the test set
y_pred = voting_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Voting Classifier Accuracy: {accuracy:.4f}")

# Score prediction
team_a_score_pred = reg_rf_team_a.predict(X_test)
team_b_score_pred = reg_rf_team_b.predict(X_test)

# Evaluate regression models with Mean Squared Error
mse_team_a = mean_squared_error(y_test_team_a, team_a_score_pred)
mse_team_b = mean_squared_error(y_test_team_b, team_b_score_pred)

print(f"Mean Squared Error for Team A's score: {mse_team_a:.4f}")
print(f"Mean Squared Error for Team B's score: {mse_team_b:.4f}")

# Save the models
joblib.dump(voting_clf, 'nba_game_predictor_voting_classifier.joblib')
joblib.dump(reg_rf_team_a, 'nba_team_a_score_predictor_rf.joblib')
joblib.dump(reg_rf_team_b, 'nba_team_b_score_predictor_rf.joblib')


