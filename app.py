from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import pandas as pd
from flask_cors import CORS

# Load the TensorFlow model and scaler
tf_model = tf.keras.models.load_model('saved_model.keras')
scaler = joblib.load('scaler.pkl')  # Ensure this file exists

# Load the dataset
df = pd.read_csv('./NHL Datasets/All Skaters 08-25.csv')  # Ensure this file exists

team_mapping = {
    "T.B": "TBL",
    "N.J": "NJD",
    "L.A": "LAK",
    "S.J": "SJS"
}
df['team'] = df['team'].replace(team_mapping)

df = df[df['situation'] == 'all']

# Define features used in TensorFlow predictions
features = ['gameScore', 'onIce_xGoalsPercentage', 'I_F_xGoals', 'I_F_primaryAssists', 'I_F_secondaryAssists',
            'I_F_shotsOnGoal', 'I_F_shotAttempts', 'I_F_goals', 'I_F_points', 'I_F_faceOffsWon', 'I_F_hits',
            'I_F_takeaways', 'I_F_giveaways', 'I_F_highDangerxGoals']

# Preprocessing logic to create necessary columns
def preprocess_data(df):
    if 'total_shots' not in df.columns:
        df['total_shots'] = df['I_F_shotsOnGoal'] + df['I_F_missedShots'] + df['I_F_blockedShotAttempts']
    
    # Remove normalization step (stats per game)
    if 'faceoffPercentage' not in df.columns:
        df['faceoffPercentage'] = df['faceoffsWon'] / df['faceoffsLost'].replace(0, 1)  # Avoid division by zero

# Identify weaknesses
def identify_weakest_areas(team_df, league_df, top_n=5):
    numeric_cols = team_df.select_dtypes(include=['number']).columns
    relevant_cols = [col for col in numeric_cols if col in features]

    team_avg = team_df[relevant_cols].mean()
    league_avg = league_df[relevant_cols].mean()

    weaknesses = (league_avg - team_avg).nlargest(top_n).index.tolist()
    return weaknesses

# Find trade target using TensorFlow model
def find_trade_target(df, team_name, weaknesses, percentile_threshold=50):
    # Filter for players in the 2024 season
    df = df[df['season'] == 2024]

    # Exclude players from the current team
    df = df[df['team'] != team_name]

    # Filter players based on weaknesses
    for weakness in weaknesses:
        if weakness in df.columns:
            threshold_value = df[weakness].quantile(percentile_threshold / 100.0)
            df = df[df[weakness] >= threshold_value]

    if df.empty:
        return []

    # Use TensorFlow model to calculate "predicted_fit"
    X_input = scaler.transform(df[features])
    df['predicted_fit'] = tf_model.predict(X_input).flatten()

    # Select top trade targets based on predicted fit
    trade_targets = df.nlargest(5, 'predicted_fit')[[
        'season', 'name', 'team', 'position', 'games_played',
        'total_shots', 'I_F_xGoals', 'gameScore', 'onIce_corsiPercentage', 'I_F_goals', 'I_F_primaryAssists', 
        'I_F_secondaryAssists', 'predicted_fit'
    ]]
    return trade_targets.to_dict(orient='records')

# Define Flask app
app = Flask(__name__)
CORS(app)

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Invalid JSON input'}), 400

    team = data.get('team', None)
    min_games_played = data.get('min_games_played', 0)

    if not team:
        return jsonify({'error': 'Please provide a team name.'}), 400

    preprocess_data(df)  # Ensure necessary columns are available
    filtered_df = df[df['games_played'] >= min_games_played]
    team_df = filtered_df[filtered_df['team'].str.upper() == team.upper()]

    if team_df.empty:
        return jsonify({'error': 'No data available for the given inputs.'}), 404

    # Identify weaknesses and find trade targets
    weakest_areas = identify_weakest_areas(team_df, filtered_df)
    trade_targets = find_trade_target(filtered_df, team.upper(), weakest_areas)

    response_data = {
        'team': team,
        'weakest_areas': weakest_areas,  # Properly formatted weaknesses
        'trade_targets': trade_targets  # Valid player recommendations
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
