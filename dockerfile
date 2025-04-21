# Utiliser une image légère Python officielle
FROM python:3.11-slim

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY api.py .
COPY modele_rand_fore/ modele_rand_fore/
COPY modeles/ modeles/
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel Uvicorn tournera
EXPOSE 8000

# Commande pour démarrer l'API FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
