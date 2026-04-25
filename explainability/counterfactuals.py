import pandas as pd
import numpy as np
import joblib
import dice_ml
from dice_ml.utils import helpers
import os

DATA_DIR = 'data'
MODEL_DIR = 'models'

def generate_counterfactuals(user_id, model_name='isolation_forest'):
    # Load data
    df = pd.read_csv(os.path.join(DATA_DIR, 'merged_features.csv'))
    X = df.drop(['user', 'is_red_team'], axis=1)
    
    # Load model
    model = joblib.load(os.path.join(MODEL_DIR, f'{model_name}.pkl'))
    
    # DiCE requires a wrapper for sklearn models
    # For anomaly detection, we treat it as a binary classification where 1 is normal and 0 is anomalous
    # However, DiCE is built for classifiers. We can wrap the decision function.
    
    def predict_fn(data):
        # Isolation Forest: decision_function > 0 is normal
        scores = model.decision_function(data)
        # Return probabilities [p_anomalous, p_normal]
        # Simple mapping: if score > 0, p_normal is high
        p_normal = 1 / (1 + np.exp(-scores))
        return np.vstack([1 - p_normal, p_normal]).T

    # Construct DiCE data and model objects
    d = dice_ml.Data(dataframe=df.drop(['user'], axis=1), continuous_features=X.columns.tolist(), outcome_name='is_red_team')
    
    # We use a custom model wrapper because anomaly detectors aren't standard classifiers
    m = dice_ml.Model(model=model, backend='sklearn', model_type='classifier')
    # Overwrite the internal predict function to use our wrapper
    m.transformer.predict_proba = predict_fn
    
    # Initialize DiCE
    exp = dice_ml.Dice(d, m, method="random")
    
    # Get the user's data
    query_instance = df[df['user'] == user_id].drop(['user', 'is_red_team'], axis=1)
    
    # Generate counterfactuals (aiming for 'is_red_team' = 0, i.e., normal)
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class=0)
    
    return dice_exp

if __name__ == "__main__":
    # Test with a known red team user
    red_users = pd.read_csv(os.path.join(DATA_DIR, 'red_team_users.csv'))
    test_user = red_users['user'].iloc[0]
    print(f"Generating counterfactuals for {test_user}...")
    cf = generate_counterfactuals(test_user)
    cf.visualize_as_dataframe()
