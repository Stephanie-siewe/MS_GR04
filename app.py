import gradio as gr
import joblib
import pandas as pd
from scipy.sparse import hstack

# Chargement des fichiers
classifier_rf = joblib.load('modele_rand_fore/classifier_rf.pkl')
vectorizer_rf = joblib.load('modeles/vectorizer_rf.pkl')

def predict_tweet(followers, following, verified, real_location, tweet_text):
    try:
        user_data = pd.DataFrame({
            'Followers': [followers],
            'Following': [following],
            'Verified': [verified],
            'Real_Location': [real_location]
        })
        text_data = vectorizer_rf.transform([tweet_text])
        combined_data = hstack((user_data.values, text_data))
        prediction = classifier_rf.predict(combined_data)
        return f"✅ Prédiction : {prediction[0]}"
    except Exception as e:
        return f"❌ Erreur : {str(e)}"

interface = gr.Interface(
    fn=predict_tweet,
    inputs=[
        gr.Number(label="Followers"),
        gr.Number(label="Following"),
        gr.Radio([0, 1], label="Verified (0 = Non, 1 = Oui)"),
        gr.Radio([0, 1], label="Real Location (0 = Non, 1 = Oui)"),
        gr.Textbox(label="Tweet Text", lines=3)
    ],
    outputs="text",
    title="Prédiction de Tweet avec Random Forest",
    description="Entrez les informations utilisateur + texte du tweet pour obtenir une prédiction."
)

if __name__ == "__main__":
    interface.launch()
