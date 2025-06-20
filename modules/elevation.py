import pandas as pd
import time
import requests 

def get_elevation(lat, lon):
    """
    Utilise l'API Open Elevation pour obtenir l'altitude d'un point donné par latitude et longitude.

    :param lat: Latitude du point
    :param lon: Longitude du point
    :return: Altitude du point en mètres
    """
    url = "https://api.open-elevation.com/api/v1/lookup"

    try:
        # Faire la demande POST à l'API pour obtenir l'altitude
        response = requests.post(url, json={"locations": [{"latitude": lat, "longitude": lon}]}, timeout=10)

        if response.status_code == 200:
            result = response.json()
            if 'results' in result and len(result['results']) > 0:
                return result['results'][0]['elevation']
            else:
                print(f"Erreur: Aucune altitude trouvée pour lat={lat}, lon={lon}")
                return None
        else:
            print(f"Erreur de connexion à l'API : {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print(f"Erreur: Délai d'attente dépassé pour lat={lat}, lon={lon}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Erreur de requête pour lat={lat}, lon={lon} : {e}")
        return None

def add_elevation_to_csv(input_file="static/_coords_converted.csv", output_file=None):
    """
    Ajoute les altitudes aux coordonnées (latitude, longitude) présentes dans un fichier CSV
    en utilisant l'API Open Elevation.

    :param input_file: Chemin vers le fichier CSV contenant les coordonnées
    :param output_file: Nom du fichier CSV de sortie. Si None, génère un fichier avec '_with_elevation'.
    """
    # Lire le fichier CSV séparé par des virgules
    df = pd.read_csv(input_file, delimiter=',')  # Assumer que le CSV est séparé par des virgules

    # Vérifier que les colonnes Latitude et Longitude existent
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        print("Les colonnes 'Latitude' et 'Longitude' sont requises dans le fichier.")
        return

    # Liste pour stocker les coordonnées avec les altitudes
    points_with_elevation = []

    # Ajouter l'altitude à chaque coordonnée
    for index, row in df.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']

        # Vérifier que les coordonnées sont valides
        if pd.notna(lat) and pd.notna(lon):
            altitude = get_elevation(lat, lon)  # Obtenir l'altitude via l'API
            if altitude is not None:
                points_with_elevation.append((lat, lon, altitude))
            else:
                print(f"Altitude non trouvée pour l'index {index}: Latitude={lat}, Longitude={lon}")
        else:
            print(f"Coordonnée invalide à l'index {index}: Latitude={lat}, Longitude={lon}")

        # Ajouter une pause de 0.1 secondes pour éviter de trop solliciter l'API (évitant le rate limit)
        time.sleep(0.1)

    # Si un fichier de sortie est spécifié, l'utiliser, sinon générer un fichier par défaut
    if output_file is None:
        # output_file = input_file.replace('.csv', '_with_elevation.csv')
        output_file = input_file.replace(input_file, 'static/coords.csv')

    # Sauvegarder les résultats dans un nouveau fichier CSV
    df_with_elevation = pd.DataFrame(points_with_elevation, columns=['Latitude', 'Longitude', 'Altitude'])
    df_with_elevation.to_csv(output_file, index=False)

    print(f"Les coordonnées avec altitude ont été sauvegardées dans '{output_file}'.")

# Exemple d'appel de la fonction :
# Demander à l'utilisateur le nom du fichier CSV d'entrée
# file = input('Saisir le nom du fichier CSV avec les coordonnées (Latitude, Longitude) : ')

# Appel de la fonction pour ajouter l'altitude
