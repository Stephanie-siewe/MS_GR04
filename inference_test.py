import joblib
from scipy.sparse import hstack
import pandas as pd

# Charger le modèle et le vectorizer
classifier_rf = joblib.load('modele_rand_fore/classifier_rf.pkl')
vectorizer_rf = joblib.load('modeles/vectorizer_rf.pkl')

# Tes nouvelles données :
new_user_features = pd.DataFrame({
    'Followers': [0],
    'Following': [1],
    'Verified': [1],
    'Real_Location': [0]
})

new_tweet_text = ["Hello, how are you ?"]

# Vectoriser le texte
new_text_tfidf = vectorizer_rf.transform(new_tweet_text)

# Combiner les features
new_features = hstack((new_user_features.values, new_text_tfidf))

# Prédire
prediction = classifier_rf.predict(new_features)
print(f"Prédiction : {prediction[0]}")
