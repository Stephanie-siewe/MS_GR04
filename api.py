from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from scipy.sparse import hstack
from dotenv import load_dotenv
import pandas as pd
import os
import psycopg2

load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cursor = conn.cursor()



def create_table():
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            followers INTEGER,
            following INTEGER,
            verified BOOLEAN,
            real_location BOOLEAN,
            tweet_text TEXT,
            prediction INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()

create_table()
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

def log_prediction(data: TweetFeatures, prediction: int):
    cursor.execute(
        """
        INSERT INTO predictions (followers, following, verified, real_location, tweet_text, prediction)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            data.followers,
            data.following,
            bool(data.verified),
            bool(data.real_location),
            data.tweet_text,
            prediction
        )
    )
    conn.commit()
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

    # Loguer la prédiction en base
    log_prediction(features, int(prediction[0]))
    
    # Retourner la réponse
    return {"prediction": int(prediction[0])}


@app.get("/logs")
def get_logs():
    cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 20")
    rows = cursor.fetchall()
    result = [
        {
            "id": row[0],
            "followers": row[1],
            "following": row[2],
            "verified": row[3],
            "real_location": row[4],
            "tweet_text": row[5],
            "prediction": row[6],
            "created_at": row[7].isoformat()
        }
        for row in rows
    ]
    return result