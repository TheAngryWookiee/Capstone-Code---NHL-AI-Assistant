import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv('./NHL Datasets/All Skaters 08-25.csv')
df_cup_data = pd.read_csv('./NHL Datasets/Stanley_Cup_Winners.csv')

team_mapping ={ 
    "T.B" : "TBL",
    "N.J" : "NJD",
    "L.A" : "LAK",
    "S.J" : "SJS" 
}

# Standardize team names in both datasets
df['team'] = df['team'].replace(team_mapping)
df_cup_data['winning_team'] = df_cup_data['winning_team'].replace(team_mapping)

# Columns of interest
columns_of_interest = [
    'season', 'name', 'team', 'position', 'situation', 'games_played', 'I_F_missedShots', 'I_F_blockedShotAttempts',
    'icetime', 'shifts', 'gameScore', 'onIce_xGoalsPercentage', 'offIce_xGoalsPercentage', 
    'onIce_corsiPercentage', 'I_F_xOnGoal', 'I_F_xGoals', 'I_F_primaryAssists', 
    'I_F_secondaryAssists', 'I_F_shotsOnGoal', 'I_F_shotAttempts', 'I_F_points', 'I_F_goals', 'I_F_savedShotsOnGoal', 
    'penalties', 'I_F_faceOffsWon', 'I_F_hits', 'I_F_takeaways', 'I_F_giveaways', 'I_F_lowDangerShots', 
    'I_F_mediumDangerShots', 'I_F_highDangerShots', 'I_F_lowDangerxGoals', 'I_F_mediumDangerxGoals', 'I_F_highDangerxGoals',
    'I_F_lowDangerGoals', 'I_F_mediumDangerGoals', 'I_F_highDangerGoals', 'I_F_dZoneGiveaways', 'I_F_oZoneShiftStarts', 
    'I_F_dZoneShiftStarts', 'I_F_neutralZoneShiftStarts', 'faceoffsWon', 'faceoffsLost', 'penaltiesDrawn', 'shotsBlockedByPlayer'
]

df = df[columns_of_interest]

# Filter for 5v5 situation
df = df[df['situation'] == '5on5']

# Handle missing values
df.fillna(0, inplace=True)

# Feature engineering
df['total_shots'] = df['I_F_shotsOnGoal'] + df['I_F_missedShots'] + df['I_F_blockedShotAttempts']

# Split data into train (<=2023) and test (2024)
df_train = df[df['season'] <= 2023]
df_test = df[df['season'] == 2024]

# Define the target variable (team success proxy) and features
features = ['gameScore', 'onIce_xGoalsPercentage', 'I_F_xGoals', 'I_F_primaryAssists', 'I_F_secondaryAssists',
            'I_F_shotsOnGoal', 'I_F_shotAttempts', 'I_F_goals', 'I_F_points', 'I_F_faceOffsWon', 'I_F_hits',
            'I_F_takeaways', 'I_F_giveaways', 'I_F_highDangerxGoals']

df_cup_winners = df_train[df_train['team'].isin(df_cup_data['winning_team'])].copy()  # Explicitly create a copy
df_cup_winners.loc[:, 'winner'] = 1
df_train = df_train.copy()
df_train.loc[:, 'winner'] = 0

# Oversample winners
df_winners_upsampled = df_cup_winners.sample(len(df_train), replace=True)

# Combine with regular teams
df_balanced = pd.concat([df_train, df_winners_upsampled])

# Redefine training data
X_train = df_balanced[features]
y_train = df_balanced['winner']

X_test = df_test[features]
y_test = df_test.get('winner', pd.Series([0] * len(df_test)))  # Default to 0 if winner column isn't present

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "TensorFlow": None  # Placeholder for TensorFlow model
}

# Define TensorFlow model
tf_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train TensorFlow model
tf_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Add TensorFlow model to models dictionary
models["TensorFlow"] = tf_model

# Train other models
for name, model in models.items():
    if name != "TensorFlow":
        model.fit(X_train, y_train)

# Identify weakest areas
def identify_weakest_areas(team_df, league_df, top_n=10):
    exclude_columns = ['icetime', 'shifts']  # Exclude non-team-relevant data
    numeric_cols = team_df.select_dtypes(include=['number']).columns
    relevant_cols = [col for col in numeric_cols if col not in exclude_columns]

    team_avg = team_df[relevant_cols].mean()
    league_avg = league_df[relevant_cols].mean()

    weaknesses = (league_avg - team_avg).nlargest(top_n).index.tolist()
    return weaknesses

# Find trade target based on model predictions
def find_trade_target(df, team_name, weaknesses, models):
    team_players = df[df['team'] == team_name]

    # Exclude superstars
    superstar_thresholds = df.groupby('team')['gameScore'].quantile(0.9)
    df = df[df['gameScore'] < df['team'].map(superstar_thresholds)]

    # Get trade candidates (players from other teams)
    trade_candidates = df[df['team'] != team_name].copy()

    # Predict fit using the model
    trade_candidates['predicted_fit'] = models["GradientBoosting"].predict(trade_candidates[features])

    # Get top 3 candidates
    top_candidates = trade_candidates.nlargest(3, 'predicted_fit')

    # Pick the player with the highest value in the weakest areas
    best_fit = top_candidates.loc[top_candidates[weaknesses].sum(axis=1).idxmax()]

    return best_fit[['name', 'team'] + weaknesses + ['gameScore', 'predicted_fit']]

selected_team = "TOR"
weakest_stats = identify_weakest_areas(df[df['team'] == selected_team], df, top_n=5)
trade_recommendation = find_trade_target(df, selected_team, weakest_stats, models)

print(f"Weakest areas for {selected_team}: {weakest_stats}")
print("Recommended trade target:")
print(trade_recommendation)