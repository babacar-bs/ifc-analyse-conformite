import pandas as pd
from pyproj import Transformer, CRS
import csv
import os
from models import *

def convert_coordinates_from_csv(input_file, espg_file=3949, espg_conversion=4326):
    """
    Convertit les coordonnées d'un fichier CSV depuis un système de projection EPSG du fichier vers un autre système EPSG.
    """
    # Définir les systèmes de projection à partir des codes EPSG
    crs_file = CRS.from_epsg(espg_file)  # EPSG du fichier d'entrée
    crs_conversion = CRS.from_epsg(espg_conversion)  # EPSG de conversion

    # Créer un transformer pour la conversion entre les deux systèmes
    transformer = Transformer.from_crs(crs_file, crs_conversion, always_xy=True)

    # Lire les coordonnées depuis le fichier CSV avec un point-virgule comme séparateur
    df = pd.read_csv(input_file, header=None, names=['X', 'Y'], delimiter=';')

    # Afficher le contenu du fichier CSV pour vérifier
    print("Contenu du fichier CSV chargé :")
    print(df)

    # Convertir les colonnes X et Y en types flottants (float)
    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')

    # Vérification des données (premières lignes)
    print("Vérification des premières lignes du fichier CSV après conversion :")
    print(df.head())

    # Conversion des points
    points_converted = []

    # Conversion de chaque point
    for index, row in df.iterrows():
        x = row['X']
        y = row['Y']

        # Vérification si les coordonnées sont valides
        if pd.notna(x) and pd.notna(y):
            lon, lat = transformer.transform(x, y)
            points_converted.append((lat, lon))
        else:
            print(f"Coordonnée invalide à l'index {index}: X={x}, Y={y}")

    # Afficher les coordonnées converties
    for lat, lon in points_converted:
        print(f"Latitude: {lat}, Longitude: {lon}")

    # # Si un fichier de sortie est spécifié, l'utiliser, sinon générer un fichier par défaut
    # if output_file is None:
    #     output_file = input_file.replace('.csv', '_converted.csv')

    # # Sauvegarder les résultats dans un nouveau fichier CSV
    # df_converted = pd.DataFrame(points_converted, columns=['Latitude', 'Longitude'])
    # df_converted.to_csv(output_file, index=False)

    df_converted = pd.DataFrame(points_converted, columns=['Latitude', 'Longitude'])

    return df_converted
# convert_coordinates_from_csv(file, espg_file, espg_conversion)


def ajouter_entete_csv(fichier_csv="static/_coords_converted.csv", entete=["Latitude", "Longitude"]):
    """
    Ajoute une ligne d'entête au début d'un fichier CSV existant.

    Args:
        fichier_csv (str): Nom du fichier CSV à modifier
        entete (list of str): Liste des noms de colonnes (par défaut ["Latitude", "Longitude"])
    """

    # Lire tout le contenu existant
    with open(fichier_csv, 'r', encoding='utf-8') as f:
        lignes = f.readlines()

    # Réécrire le fichier avec l'entête au début
    with open(fichier_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(entete)
        for ligne in lignes:
            f.write(ligne)

    print(f"Entête ajoutée à {fichier_csv}")




# def transformer_fichier_parcelle(analysis_id, output_folder="outputs"):#(input_csv="outputs/coords_dessin_polygone.csv", output_folder="outputs"):
#     # Lire le CSV d'origine
#     #df = pd.read_csv(input_csv)
#     # Récupérer l'analyse depuis la base
#     analysis = Analysis.query.get(analysis_id)
#     if not analysis or not analysis.parcelle_data:
#         print("❌ Analyse ou données de parcelle introuvables")
#         return None

#     # Convertir les données JSON en DataFrame
#     try:
#         points_data = analysis.parcelle_data.get('points', [])
#         df = pd.DataFrame(points_data)
        
#         # Renommer les colonnes pour compatibilité avec le code existant
#         df = df.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'})
        
#         # Vérification des données
#         if df.empty or 'X' not in df.columns or 'Y' not in df.columns:
#             print("❌ Structure de données invalide")
#             return None

#     except Exception as e:
#         print(f"❌ Erreur de conversion des données: {str(e)}")
#         return None

#     # Créer le dossier de sortie s'il n'existe pas
#     os.makedirs(output_folder, exist_ok=True)

#     # Grouper par id_parcelle
#     grouped = df.groupby('id_parcelle')

#     for parcelle_id, group in grouped:
#         # Latitude = X, Longitude = Y
#         transformed = group[['X', 'Y']].rename(columns={'X': 'Latitude', 'Y': 'Longitude'})

#         # # Nom du fichier de sortie
#         # output_file = os.path.join(output_folder, f"parcelle_{parcelle_id}.csv")
#         # # Écriture du fichier CSV
#         # transformed.to_csv(output_file, index=False)
#         # print(f"✅ Fichier créé : {output_file}")


def transformer_fichier_parcelle(analysis_id):
    """
    Transforme les données de parcelle et les stocke en JSON dans la base de données
    au lieu de générer des fichiers CSV.
    
    Args:
        analysis_id (int): ID de l'analyse à traiter
        
    Returns:
        bool: True si succès, False si échec
    """
    # Récupérer l'analyse depuis la base
    analysis = Analysis.query.get(analysis_id)
    if not analysis or not analysis.parcelle_data:
        print("❌ Analyse ou données de parcelle introuvables")
        return False

    try:
        # Convertir les données JSON en DataFrame
        points_data = analysis.parcelle_data.get('points', [])
        df = pd.DataFrame(points_data)
        
        # Renommer les colonnes
        df = df.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'})
        
        # Vérification des données
        if df.empty or 'X' not in df.columns or 'Y' not in df.columns:
            print("❌ Structure de données invalide")
            return False

        # Initialiser la structure de stockage
        parcelle_coords = {}

        # Grouper et traiter chaque parcelle
        for parcelle_id, group in df.groupby('id_parcelle'):
            parcelle_coords[str(parcelle_id)] = {
                'coordinates': group[['X', 'Y']]
                    .rename(columns={'X': 'Latitude', 'Y': 'Longitude'})
                    .to_dict(orient='records'),
            }

        # Mise à jour de l'analyse
        analysis.parcelle_coords = parcelle_coords
        db.session.commit()
        
        print(f"✅ Données de {len(parcelle_coords)} parcelles sauvegardées dans l'analyse {analysis_id}")
        return True

    except Exception as e:
        db.session.rollback()
        print(f"❌ Erreur lors du traitement: {str(e)}")
        return False