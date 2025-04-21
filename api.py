from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from scipy.sparse import hstack
import pandas as pd

# Créer l'app FastAPI
app = FastAPI()

# Charger modèle et vectorizer au démarrage
classifier_rf = joblib.load('modele_rand_fore/classifier_rf.pkl')
vectorizer_rf = joblib.load('modeles/vectorizer_rf.pkl')

# Définir le format attendu pour les requêtes
class TweetFeatures(BaseModel):
    followers: int
    following: int
    verified: int
    real_location: int
    tweet_text: str

# Route principale pour faire une prédiction
@app.post("/predict")
def predict(features: TweetFeatures):
    # Préparer les features utilisateur
    user_data = pd.DataFrame({
        'Followers': [features.followers],
        'Following': [features.following],
        'Verified': [features.verified],
        'Real_Location': [features.real_location]
    })
    
    # Vectoriser le texte
    text_data = vectorizer_rf.transform([features.tweet_text])
    
    # Combiner user + texte
    combined_data = hstack((user_data.values, text_data))
    
    # Faire la prédiction
    prediction = classifier_rf.predict(combined_data)
    
    # Retourner la réponse
    return {"prediction": int(prediction[0])}
