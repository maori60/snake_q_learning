import numpy as np
import pandas as pd

# Charger les données depuis le fichier CSV
data = pd.read_csv("data.csv")
X = data["km"].values
Y = data["price"].values
m = len(X)

# Normalisation des données pour un meilleur entraînement
X_min = X.min()
X_max = X.max()
X_norm = (X - X_min) / (X_max - X_min)

# Initialisation des paramètres
theta0 = 0
theta1 = 0
learning_rate = 0.1
epochs = 10000

# Descente de gradient pour entraîner le modèle
for _ in range(epochs):
    predicted_price = theta0 + theta1 * X_norm
    error = predicted_price - Y
    theta0 -= learning_rate * np.sum(error) / m
    theta1 -= learning_rate * np.sum(error * X_norm) / m

def predict_price(mileage_input, theta0, theta1, X_min, X_max, min_price=1000):
    """
    Fonction qui prédit le prix d'une voiture en fonction de son kilométrage.
    """
    mileage_norm = (mileage_input - X_min) / (X_max - X_min)
    estimated_price = theta0 + theta1 * mileage_norm
    return max(min_price, estimated_price)  # Assure un prix minimum réaliste

# Programme interactif
while True:
    mileage_input = input("Entrez le kilométrage de votre voiture (ou 'exit' pour quitter) : ")
    if mileage_input.lower() == 'exit':
        break
    try:
        mileage_input = float(mileage_input)
        estimated_price = predict_price(mileage_input, theta0, theta1, X_min, X_max)
        print(f"\n💰 Prix estimé pour une voiture avec {mileage_input} km : {estimated_price:.2f} €\n")
    except ValueError:
        print("\n⛔ Veuillez entrer un nombre valide.\n")
