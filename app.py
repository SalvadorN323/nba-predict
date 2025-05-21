import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from dotenv import load_dotenv
import os


load_dotenv()
   
   
  
app = Flask(__name__)
CORS(app, origins=['https://nba-predict-1.onrender.com'])


openai.api_key = os.getenv('OPENAI_API_KEY')


#load the pre-trained xgboost model
model = joblib.load('model.pkl')

#load the CSV data (replace the file path if needed)
df = pd.read_csv('nba_games.csv')

#ensure the 'GAME_DATE' column is in datetime format for sorting
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        #extract data from the request
        data = request.get_json()

        team_a = data.get('team_a')
        team_b = data.get('team_b')

        if not team_a or not team_b:
            return jsonify({'error': 'Both teams must be selected'})

        #filter data to get the most recent game for each team
        team_a_data = df[df['TEAM_NAME'] == team_a].sort_values(by='GAME_DATE', ascending=False).head(1)
        team_b_data = df[df['TEAM_NAME'] == team_b].sort_values(by='GAME_DATE', ascending=False).head(1)

        #if any of the teams does not have a recent game, return an error
        if team_a_data.empty or team_b_data.empty:
            return jsonify({'error': f'No data found for {team_a} or {team_b}'})

        #extract relevant stats for both teams from the most recent game
        team_a_stats = team_a_data[['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB']].iloc[0]
        team_b_stats = team_b_data[['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB']].iloc[0]

        #prepare the data for prediction using only the features the model was trained on
        matchup_data = {
            'PTS': float(team_a_stats['PTS']),
            'FGM': float(team_a_stats['FGM']),
            'FGA': float(team_a_stats['FGA']),
            'FG_PCT': float(team_a_stats['FG_PCT']),
            'FG3M': float(team_a_stats['FG3M']),
            'FG3A': float(team_a_stats['FG3A']),
            'FG3_PCT': float(team_a_stats['FG3_PCT']),
            'FTM': float(team_a_stats['FTM']),
            'FTA': float(team_a_stats['FTA']),
            'REB': float(team_a_stats['REB']),
            'AST': float(team_a_stats['AST']),
            'TOV': float(team_a_stats['TOV']),
            'STL': float(team_a_stats['STL']),
            'BLK': float(team_a_stats['BLK']),
            'OREB': float(team_a_stats['OREB']),
            'DREB': float(team_a_stats['DREB'])
        }

        #convert the data into a DataFrame for the model
        game_data = pd.DataFrame([matchup_data])

        #make the classification prediction using the trained model
        prediction = model.predict(game_data)  # Final win/loss prediction
        prediction_probabilities = model.predict_proba(game_data)  # Probabilities for win/loss

        #extract the win probability (probability for class 1)
        win_probability = float(prediction_probabilities[0, 1])  # Convert to native float
        loss_probability = float(prediction_probabilities[0, 0])  # Convert to native float

        #convert the prediction result
        result = "Team A wins!" if prediction[0] == 1 else "Team B wins!"
        
        #also return the graph data (stats comparison)
        graph_data = {
            "labels": ['PTS', 'FG_PCT', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB'],
            "team_a": [float(matchup_data['PTS']), float(matchup_data['FG_PCT']), float(matchup_data['REB']), float(matchup_data['AST']), float(matchup_data['TOV']), float(matchup_data['STL']), float(matchup_data['BLK']), float(matchup_data['OREB']), float(matchup_data['DREB'])],
            "team_b": [float(team_b_stats['PTS']), float(team_b_stats['FG_PCT']), float(team_b_stats['REB']), float(team_b_stats['AST']), float(team_b_stats['TOV']), float(team_b_stats['STL']), float(team_b_stats['BLK']), float(team_b_stats['OREB']), float(team_b_stats['DREB'])]
        }

        #call OpenAI API for analysis
        prompt = f"""
        Analyze the following NBA game prediction:
        Team A: {team_a} 
        Team B: {team_b}
        Stats: {matchup_data}
        Model Prediction: {result}
        Win Probability: {win_probability:.2f}
        Loss Probability: {loss_probability:.2f}
        You are an expert NBA analyst so provide a short paragraph explanation of the prediction based on the stats differences and probabilities.
        Provide the current season stats for both teams and any other relevant information and end off why you think the model prediction is correct.
        Provide links to any relevant sources.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        gpt_analysis = response['choices'][0]['message']['content'].strip()

        return jsonify({
            'prediction': result,
            'win_probability': win_probability,
            'loss_probability': loss_probability,
            'graph_data': graph_data,
            'analysis': gpt_analysis
        })

    except Exception as e:
        print(f"Error: {str(e)}")  #this will be visible in the server logs
        return jsonify({'error': f"An error occurred: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port) 