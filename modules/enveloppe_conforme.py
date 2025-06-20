import csv
import math
import time
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyproj import Transformer
from shapely.geometry import Point, LineString, Polygon
import os
from modules.conversion import transformer_fichier_parcelle
from models import *

# hauteur = 2

# # === PARAMÈTRES DES RÈGLES URBANISTIQUES
# regles = {
#     'VO': {
#         'hauteur_max': hauteur, #24,
#         'retrait': 2,
#         'angle_prospect': None,  # Pas d'angle de prospect pour les voies
#         'retrait_min': 2
#     },
#     'LS': {
#         'hauteur_max': hauteur ,#21,
#         'retrait': 5,
#         'angle_prospect': 'H/2',  # L=H/2 pour les limites séparatives
#         'retrait_min': {
#             'aveugle': 2.5,       # Mur aveugle: minimum 2.5m
#             'fenetre': 5          # Mur avec fenêtre: minimum 5m
#         }
#     },
#     'FP': {
#         'hauteur_max': hauteur, #21,
#         'retrait': 5,
#         'angle_prospect': 'H/2',  # L=H/2 pour les fonds de parcelle
#         'retrait_min': {
#             'aveugle': 2.5,       # Mur aveugle: minimum 2.5m
#             'fenetre': 5          # Mur avec fenêtre: minimum 5m
#         }
#     },
#     'BB': {  # Règle entre bâtiments
#         'angle_prospect': 'H',    # L=H entre bâtiments
#         'retrait_min': {
#             'aveugle': 2.5,       # Si mur aveugle: H/2 avec minimum 2.5m
#             'fenetre': 5          # Si mur avec fenêtre: H avec minimum 5m
#         }
#     }
# }


regles: Dict[str, Dict[str, Any]] = {}

def init_regles(hauteur: float) -> None:
    global regles
    regles = {
        'VO': {
            'hauteur_max': min(24, hauteur),
            'retrait': 0,#2,
            'angle_prospect': None,
            'retrait_min': 0
        },
        'LS': {
            'hauteur_max': min(21, hauteur),
            'retrait': 5,
            'angle_prospect': hauteur / 2,
            'retrait_min': {
                'aveugle': max(2.5, hauteur / 2),
                'fenetre': max(5, hauteur)
            }
        },
        'FP': {
            'hauteur_max': min(21, hauteur),
            'retrait': 5,
            'angle_prospect': hauteur / 2,
            'retrait_min': {
                'aveugle': max(2.5, hauteur / 2),
                'fenetre': max(5, hauteur)
            }
        },
        'BB': {
            'angle_prospect': hauteur,
            'retrait_min': {
                'aveugle': max(2.5, hauteur / 2),
                'fenetre': max(5, hauteur)
            }
        }
    }




# === LIRE CSV GPS
# def lire_points_csv(fichier="static/_coords_converted.csv"):
#     points = []
#     with open(fichier, newline='') as csvfile:
#         reader = csv.DictReader(csvfile, delimiter=',')
#         for row in reader:
#             try:
#                 lon = float(row['Longitude'].replace(',', '.'))
#                 lat = float(row['Latitude'].replace(',', '.'))
#                 alt = float(row.get('Altitude', '0').replace(',', '.'))  # Prend 0 si non fourni
#                 points.append((lon, lat, alt))
#             except ValueError:
#                 continue
#     return points
def lire_points_csv(analysis_id):
    """
    Lit les coordonnées des points à partir de la base de données (colonne parcelle_coords).
    
    Args:
        analysis_id (int): ID de l'analyse à lire.
    
    Returns:
        list of tuples: Liste de tuples (Longitude, Latitude, Altitude)
    """
    analysis = Analysis.query.get(analysis_id)
    if not analysis or not analysis.parcelle_coords:
        print("❌ Analyse introuvable ou pas de données de parcelle")
        return []

    points = []
    try:
        for parcelle in analysis.parcelle_coords.values():
            for point in parcelle['coordinates']:
                try:
                    lat = float(point['Latitude'])
                    lon = float(point['Longitude'])
                    alt = float(point.get('Altitude', 0))  # facultatif
                    points.append((lon, lat, alt))
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"❌ Erreur lors de la lecture des points : {str(e)}")
        return []

    return points


# === CONVERTIR EN LAMBERT93
def convertir_en_lambert93(points_gps):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    return [(x, y, z) for (x, y, z) in [(*transformer.transform(lon, lat), alt) for lon, lat, alt in points_gps]]


# === CALCULER LA NORMALE D'UN SEGMENT (ORIENTÉE VERS L'INTÉRIEUR)
def calculer_normale(point1, point2, points_polygone):
    """Version robuste avec gestion des points identiques"""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]

    # Tolérance réaliste (1 cm)
    if math.sqrt(dx*dx + dy*dy) < 0.01:
        print(f"Avertissement : Points quasi-identiques entre {point1} et {point2}")
        # Retourne une normale par défaut (1,0) avec longueur normalisée
        return (1.0, 0.0)

    # Calcul des normales possibles
    length = math.sqrt(dx*dx + dy*dy)
    nx1, ny1 = -dy/length, dx/length  # Normale 1
    nx2, ny2 = dy/length, -dx/length  # Normale 2

    # Vérification de l'orientation
    poly = Polygon(points_polygone)
    centroid = poly.centroid
    test_vector = (centroid.x - (point1[0]+point2[0])/2,
                   centroid.y - (point1[1]+point2[1])/2)

    # Produit scalaire pour déterminer la bonne normale
    if (nx1 * test_vector[0] + ny1 * test_vector[1]) > 0:
        return (float(nx1), float(ny1))  # Conversion explicite en float
    else:
        return (float(nx2), float(ny2))  # Conversion explicite en float

# === DÉTERMINER LE RETRAIT RÉGLEMENTAIRE EN FONCTION DU TYPE DE SEGMENT ET DE MUR
def determiner_retrait(type_segment, type_mur, regles):
    """
    Calcule le retrait réglementaire en fonction du type de segment et de mur.
    """
    regle = regles[type_segment]

    # Pour les voies (VO), le retrait est fixe
    if type_segment == 'VO':
        return regle['retrait']

    # Pour les limites séparatives (LS) et fonds de parcelle (FP)
    # le retrait dépend du type de mur (aveugle ou avec fenêtres)
    retrait_min = regle['retrait_min'][type_mur]

    # Le retrait standard spécifié dans les règles est comparé au minimum
    return max(regle['retrait'], retrait_min)

# === DÉCALER UN SEGMENT EN PARALLÈLE
def decaler_segment_parallele(point1, point2, normale, retrait):
    """Version sécurisée avec vérification des types"""
    # Vérification du type de la normale
    if not isinstance(normale, (tuple, list)) or len(normale) != 2:
        raise TypeError(f"La normale doit être un tuple de 2 floats, reçu: {normale} (type: {type(normale)})")

    try:
        nx = float(normale[0])
        ny = float(normale[1])
        retrait = float(retrait)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Impossible de convertir les valeurs en float: {e}")

    # Calcul des points décalés
    point1_decale = (float(point1[0]) + nx * retrait,
                    float(point1[1]) + ny * retrait)
    point2_decale = (float(point2[0]) + nx * retrait,
                    float(point2[1]) + ny * retrait)

    return point1_decale, point2_decale



def decaler_segments(points, types_segments, types_murs, regles):
    """
    Version simplifiée qui utilise les retraits calculés
    """
    # 1. Calcul des retraits
    retraits = determiner_retraits(types_segments, types_murs, regles)

    # 2. Application des retraits
    segments_decales = []
    points_2d = [(p[0], p[1]) for p in points]

    for i in range(len(points)):
        p1 = points_2d[i]
        p2 = points_2d[(i+1)%len(points_2d)]
        normale = calculer_normale(p1, p2, points_2d)
        seg_decale = decaler_segment_parallele(p1, p2, normale, retraits[i])
        segments_decales.append(seg_decale)

    return segments_decales, retraits  # Retourne aussi les retraits pour vérification


# === TROUVER L'INTERSECTION ENTRE DEUX LIGNES
def trouver_intersection(ligne1, ligne2):
    """
    Trouve le point d'intersection entre deux lignes.
    Retourne None s'il n'y a pas d'intersection.
    """
    line1 = LineString(ligne1)
    line2 = LineString(ligne2)

    if line1.intersects(line2):
        intersection = line1.intersection(line2)
        if intersection.geom_type == 'Point':
            return (intersection.x, intersection.y)

    return None

# === CONNECTER LES SEGMENTS DÉCALÉS
def connecter_segments(segments_decales):
    """
    Connecte les segments décalés pour former un polygone fermé.
    Utilise les intersections quand elles existent, sinon crée des segments de connexion.
    """
    nb_segments = len(segments_decales)
    points_enveloppe = []

    for i in range(nb_segments):
        segment_actuel = segments_decales[i]
        segment_suivant = segments_decales[(i+1) % nb_segments]

        # Chercher l'intersection entre le segment actuel et le suivant
        intersection = trouver_intersection(segment_actuel, segment_suivant)

        if intersection:
            # Si une intersection existe, utiliser ce point comme connexion
            points_enveloppe.append(intersection)
        else:
            # Si pas d'intersection, utiliser une approximation
            # Calculer le point milieu entre la fin du segment actuel et le début du suivant
            fin_actuel = segment_actuel[1]
            debut_suivant = segment_suivant[0]
            milieu = ((fin_actuel[0] + debut_suivant[0])/2, (fin_actuel[1] + debut_suivant[1])/2)
            points_enveloppe.append(milieu)

    return points_enveloppe

# === GÉNÉRER L'ENVELOPPE RÉGLEMENTAIRE COMPLÈTE
def generer_enveloppe_reglementaire(points, types_segments, types_murs, regles):
    """
    Génère l'enveloppe réglementaire complète en:
    1. Décalant chaque segment en parallèle selon son retrait
    2. Connectant les segments décalés pour former un polygone fermé
    """
    # Décaler les segments en parallèle
    # segments_decales = decaler_segments(points, types_segments, types_murs, regles)
    segments_decales, retraits = decaler_segments(points, types_segments, types_murs, regles)


    # Connecter les segments décalés
    points_enveloppe = connecter_segments(segments_decales)

    return segments_decales, points_enveloppe


def determiner_retraits(types_segments, types_murs, regles):
    """
    Calcule UNIQUEMENT les retraits nécessaires à partir des règles.
    Retourne une liste de distances L pour chaque segment.
    """
    retraits = []
    for i, typ in enumerate(types_segments):
        regle = regles[typ]

        if typ == 'VO':
            # Voie : retrait fixe
            retraits.append(regle['retrait'])
        else:
            # Règles LS/FP
            H_max = regle['hauteur_max']  # Hauteur fixe du PLU (ex: 21m)

            # Calcul théorique du retrait
            if regle['angle_prospect'] == 'H/2':
                L_theorique = H_max / 2  # L = H/2
            else:  # 'H'
                L_theorique = H_max       # L = H

            # Application des minima
            type_mur = types_murs[i] if types_murs else 'aveugle'
            L_min = regle['retrait_min'][type_mur]

            retraits.append(max(L_theorique, L_min))
    return retraits


# === NOUVELLES FONCTIONS POUR GESTION 3D ===


def visualiser_3d_maillage_coherent(analyse_id, points_enveloppe, hauteurs_segments, chemin_fichier="static/enveloppe_3d.png"):
    """
    Génère une visualisation 3D de l'enveloppe réglementaire et enregistre l'image dans un fichier PNG.

    Arguments :
        points_enveloppe : liste des sommets [(x, y)]
        hauteurs_segments : liste des hauteurs associées à chaque segment
        chemin_fichier : chemin du fichier image de sortie
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    faces = []
    colors = []
    cmap = plt.get_cmap('tab10')

    for i in range(len(points_enveloppe)):
        p1 = points_enveloppe[i]
        p2 = points_enveloppe[(i+1) % len(points_enveloppe)]
        hauteur = hauteurs_segments[i]

        pts = [
            [p1[0], p1[1], 0],
            [p2[0], p2[1], 0],
            [p2[0], p2[1], hauteur],
            [p1[0], p1[1], hauteur]
        ]

        faces.append(pts)
        colors.append(cmap(i % 10))

    # Ajouter le plancher
    plancher_pts = [[p[0], p[1], 0] for p in points_enveloppe]
    faces.append(plancher_pts)
    colors.append('lightgray')

    # Ajouter un toit (hauteur minimale)
    hauteur_min = min(hauteurs_segments)
    toit_pts = [[p[0], p[1], hauteur_min] for p in points_enveloppe]
    faces.append(toit_pts)
    colors.append('darkgray')

    poly = Poly3DCollection(faces, alpha=0.7)
    poly.set_facecolor(colors)
    poly.set_edgecolor('black')
    ax.add_collection3d(poly)

    xs = [p[0] for face in faces for p in face]
    ys = [p[1] for face in faces for p in face]
    zs = [p[2] for face in faces for p in face]

    ax.set_xlim([min(xs), max(xs)])
    ax.set_ylim([min(ys), max(ys)])
    ax.set_zlim([0, max(zs) * 1.1])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Hauteur (m)')
    ax.set_title('Enveloppe réglementaire 3D - Maillage cohérent')

    for i in range(len(points_enveloppe)):
        p1 = points_enveloppe[i]
        p2 = points_enveloppe[(i+1) % len(points_enveloppe)]
        hauteur = hauteurs_segments[i]
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, hauteur / 2, f"S{i}\nh={hauteur:.1f}m", color='black', fontsize=8, ha='center')

    ax.view_init(elev=30, azim=45)

    # Sauvegarder dans un fichier PNG
     # Créer le dossier dédié


    analysis = Analysis.query.get(analyse_id)
    analysis_dir = "static/outputs/analyse_" + str(analysis.user_id) + "_"+ str(analyse_id) + "/"#os.path.join('static', 'analyses', str(analysis_id))
    os.makedirs(analysis_dir, exist_ok=True)
    print("Dossier créé")

    chemin = analysis_dir + "envellope_3d.png"
    plt.tight_layout()
    plt.savefig(chemin, dpi=150, format='png')
    # analysis = Analysis.query.get(analyse_id)
    if not analysis.enveloppe3d_img:
        analysis.enveloppe3d_img = {}
    # S'assurer que c'est bien un dictionnaire, au cas où la DB contient une chaîne JSON par exemple
    if isinstance(analysis.enveloppe3d_img, str):
        import json
        analysis.enveloppe3d_img = json.loads(analysis.enveloppe3d_img)
    analysis.enveloppe3d_img['enveloppe_3d'] = chemin
    db.session.commit()

    # plt.savefig(chemin_fichier, dpi=150, format='png')
    plt.close(fig)  # Fermer la figure pour libérer la mémoire
    time.sleep(0.2)

    return chemin_fichier  # On retourne le chemin du fichier image


# === VISUALISER LES PLANS DE PROSPECT
def visualiser_plans_prospect(points_orig, points_enveloppe, hauteurs, types_segments, types_murs):
    """
    Visualise les plans obliques de prospect qui s'appliquent à chaque segment.
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Tracer le polygone original au sol
    xs_orig = [p[0] for p in points_orig]
    ys_orig = [p[1] for p in points_orig]
    xs_orig.append(points_orig[0][0])
    ys_orig.append(points_orig[0][1])
    zs_orig = [0] * (len(points_orig) + 1)
    ax.plot(xs_orig, ys_orig, zs_orig, 'b-', linewidth=2, label="Parcelle originale")

    # Tracer l'enveloppe au sol
    xs_env = [p[0] for p in points_enveloppe]
    ys_env = [p[1] for p in points_enveloppe]
    xs_env.append(points_enveloppe[0][0])
    ys_env.append(points_enveloppe[0][1])
    zs_env = [0] * (len(points_enveloppe) + 1)
    ax.plot(xs_env, ys_env, zs_env, 'r-', linewidth=2, label="Enveloppe réglementaire")

    # Tracer l'enveloppe en hauteur
    for i in range(len(points_enveloppe)):
        p1 = points_enveloppe[i]
        p2 = points_enveloppe[(i+1) % len(points_enveloppe)]
        hauteur = hauteurs[i]

        # Tracer le segment vertical à chaque point
        ax.plot([p1[0], p1[0]], [p1[1], p1[1]], [0, hauteur], 'r-', alpha=0.5)

        # Tracer le segment horizontal en hauteur
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [hauteur, hauteur], 'r-', alpha=0.7)

        # Afficher le type de mur et les règles
        milieu_x = (p1[0] + p2[0]) / 2
        milieu_y = (p1[1] + p2[1]) / 2

        # Trouve le segment original correspondant
        # (approximation: prendre le plus proche en distance)
        min_dist = float('inf')
        idx_original = 0
        segment_env = LineString([p1, p2])

        for j in range(len(points_orig)):
            orig_p1 = (points_orig[j][0], points_orig[j][1])
            orig_p2 = (points_orig[(j+1) % len(points_orig)][0], points_orig[(j+1) % len(points_orig)][1])
            segment_orig = LineString([orig_p1, orig_p2])
            dist = segment_env.distance(segment_orig)

            if dist < min_dist:
                min_dist = dist
                idx_original = j

        # Récupérer les informations du type de segment et du mur
        type_segment = types_segments[idx_original] if idx_original < len(types_segments) else "?"
        type_mur = types_murs[idx_original] if idx_original < len(types_murs) else "?"

        # Afficher les infos
        label = f"S{i} ({type_segment}"
        if type_mur is not None:
            label += f", {type_mur}"
        label += f")\nh={hauteur:.1f}m"

        ax.text(milieu_x, milieu_y, hauteur, label,
                color='black', fontsize=8, ha='center', va='bottom')

    # Tracer les plans obliques de prospect pour chaque segment
    for i in range(len(points_enveloppe)):
        # Trouver le segment original correspondant
        min_dist = float('inf')
        idx_original = 0
        p1 = points_enveloppe[i]
        p2 = points_enveloppe[(i+1) % len(points_enveloppe)]
        segment_env = LineString([p1, p2])

        for j in range(len(points_orig)):
            orig_p1 = (points_orig[j][0], points_orig[j][1])
            orig_p2 = (points_orig[(j+1) % len(points_orig)][0], points_orig[(j+1) % len(points_orig)][1])
            segment_orig = LineString([orig_p1, orig_p2])
            dist = segment_env.distance(segment_orig)

            if dist < min_dist:
                min_dist = dist
                idx_original = j

        type_segment = types_segments[idx_original] if idx_original < len(types_segments) else "?"
        type_mur = types_murs[idx_original] if idx_original < len(types_murs) else "?"

        if type_segment in ['LS', 'FP', 'BB']:
            # Tracer le plan oblique de prospect
            # Récupérer la règle de prospect
            regle = regles[type_segment]
            angle_prospect = regle['angle_prospect']

            if angle_prospect:
                # Générer un maillage pour visualiser le plan
                # On crée un quadrillage à partir des segments original et décalé
                orig_p1 = (points_orig[idx_original][0], points_orig[idx_original][1])
                orig_p2 = (points_orig[(idx_original+1) % len(points_orig)][0], points_orig[(idx_original+1) % len(points_orig)][1])

                # Direction du segment
                vec_seg = np.array([orig_p2[0] - orig_p1[0], orig_p2[1] - orig_p1[1]])
                longueur_seg = np.linalg.norm(vec_seg)
                vec_seg_norm = vec_seg / longueur_seg if longueur_seg > 0 else np.array([0, 0])

                # Points pour créer la grille du plan oblique
                nb_points = 10
                points_grille_x = []
                points_grille_y = []
                points_grille_z = []

                # Calculer l'équation du plan oblique basé sur le prospect
                if angle_prospect == 'H/2':  # H = 2L
                    for t in np.linspace(0, 1, nb_points):
                        # Point le long du segment original
                        pt_orig_x = orig_p1[0] + t * vec_seg[0]
                        pt_orig_y = orig_p1[1] + t * vec_seg[1]

                        # Point correspondant sur le segment d'enveloppe
                        pt_env_x = p1[0] + t * (p2[0] - p1[0])
                        pt_env_y = p1[1] + t * (p2[1] - p1[1])

                        # Distance entre les deux points
                        dx = pt_env_x - pt_orig_x
                        dy = pt_env_y - pt_orig_y
                        dist = np.sqrt(dx**2 + dy**2)

                        # Hauteur au point de l'enveloppe selon la règle H=2L
                        hauteur_max = min(2 * dist, hauteurs[i])

                        # Ajouter plusieurs points pour créer le plan
                        for s in np.linspace(0, 1, 5):
                            h = s * hauteur_max
                            points_grille_x.append(pt_env_x)
                            points_grille_y.append(pt_env_y)
                            points_grille_z.append(h)

                    # Tracer le plan oblique
                    surf = ax.plot_trisurf(np.array(points_grille_x), np.array(points_grille_y), np.array(points_grille_z),
                                          color='cyan', alpha=0.3)

                elif angle_prospect == 'H':  # H = L
                    for t in np.linspace(0, 1, nb_points):
                        # Point le long du segment original
                        pt_orig_x = orig_p1[0] + t * vec_seg[0]
                        pt_orig_y = orig_p1[1] + t * vec_seg[1]

                        # Point correspondant sur le segment d'enveloppe
                        pt_env_x = p1[0] + t * (p2[0] - p1[0])
                        pt_env_y = p1[1] + t * (p2[1] - p1[1])

                        # Distance entre les deux points
                        dx = pt_env_x - pt_orig_x
                        dy = pt_env_y - pt_orig_y
                        dist = np.sqrt(dx**2 + dy**2)

                        # Hauteur au point de l'enveloppe selon la règle H=L
                        hauteur_max = min(dist, hauteurs[i])

                        # Ajouter plusieurs points pour créer le plan
                        for s in np.linspace(0, 1, 5):
                            h = s * hauteur_max
                            points_grille_x.append(pt_env_x)
                            points_grille_y.append(pt_env_y)
                            points_grille_z.append(h)

                    # Tracer le plan oblique
                    surf = ax.plot_trisurf(np.array(points_grille_x), np.array(points_grille_y), np.array(points_grille_z),
                                          color='magenta', alpha=0.3)

    # Configuration de l'affichage
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Hauteur (m)')
    ax.set_title('Plans de prospect - Visualisation 3D')

    # Améliorer la vue 3D
    ax.view_init(elev=30, azim=45)

    # Ajouter une légende
    ax.legend()

    plt.tight_layout()
    plt.show()

    return fig, ax


# def exporter_points_csv(points, hauteurs, nom_fichier="static/enveloppe_3d.csv", mode="enveloppe"):
#     """
#     Exporte les données géométriques et les hauteurs dans un fichier CSV structuré.

#     Args:
#         points: Liste de points (mode 'enveloppe') ou de segments (mode 'segments')
#         hauteurs: Liste des hauteurs maximales autorisées
#         nom_fichier: Chemin du fichier de sortie
#         mode: 'enveloppe' (points connectés) ou 'segments' (segments individuels)

#     """
#     with open(nom_fichier, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)

#         # Entête adapté au mode
#         if mode == "enveloppe":
#             writer.writerow(['Segment', 'X1', 'Y1', 'X2', 'Y2', 'Hauteur'])
#             for i in range(len(points)):
#                 p1 = points[i]
#                 p2 = points[(i+1) % len(points)]
#                 writer.writerow([i, p1[0], p1[1], p2[0], p2[1], hauteurs[i]])
#         else:
#             writer.writerow(['Segment_ID', 'X1', 'Y1', 'X2', 'Y2', 'Hauteur'])
#             for i, (p1, p2) in enumerate(points):
#                 writer.writerow([i, p1[0], p1[1], p2[0], p2[1], hauteurs[i]])

#     print(f"Export réussi : {nom_fichier} ({len(points)} éléments)")

def exporter_points_csv(analysis_id, points, hauteurs, mode="enveloppe"):
    """
    Exporte les données géométriques et les hauteurs dans la base de données
    dans la colonne 'enveloppe_finale' (au format JSON).
    
    Args:
        analysis_id (int): ID de l'analyse concernée.
        points: Liste de points ou de segments.
        hauteurs: Liste des hauteurs maximales autorisées.
        mode: 'enveloppe' (points connectés) ou 'segments' (segments individuels).
    """
    analysis = Analysis.query.get(analysis_id)
    if not analysis:
        print(f"❌ Analyse {analysis_id} introuvable")
        return False

    enveloppe_data = []

    try:
        if mode == "enveloppe":
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                enveloppe_data.append({
                    "segment": i,
                    "X1": p1[0],
                    "Y1": p1[1],
                    "X2": p2[0],
                    "Y2": p2[1],
                    "hauteur": hauteurs[i]
                })
        else:  # mode "segments"
            for i, (p1, p2) in enumerate(points):
                enveloppe_data.append({
                    "segment": i,
                    "X1": p1[0],
                    "Y1": p1[1],
                    "X2": p2[0],
                    "Y2": p2[1],
                    "hauteur": hauteurs[i]
                })

        # Enregistrement dans la base de données
        analysis.enveloppe_finale = enveloppe_data
        db.session.commit()
        print(f"✅ Export réussi vers la base pour l'analyse {analysis_id} ({len(enveloppe_data)} segments)")
        return True

    except Exception as e:
        db.session.rollback()
        print(f"❌ Erreur lors de l'export vers la base : {str(e)}")
        return False



# === FONCTION POUR CALCULER LA SURFACE DE PLANCHER MAXIMALE ===
def calculer_surface_plancher_max(points_enveloppe, hauteurs_segments):
    """
    Calcule une estimation de la surface de plancher maximale possible
    dans l'enveloppe réglementaire, en considérant un nombre d'étages basé sur les hauteurs.
    """
    # Créer un polygone avec les points de l'enveloppe
    polygon = Polygon([(p[0], p[1]) for p in points_enveloppe])

    # Calculer la surface de l'emprise au sol
    surface_emprise = polygon.area

    # Hauteur moyenne des segments (pour estimer le nombre d'étages)
    hauteur_moyenne = sum(hauteurs_segments) / len(hauteurs_segments)

    # Estimer le nombre d'étages (en considérant environ 3m par étage)
    nb_etages_max = int(hauteur_moyenne / 3)

    # Surface de plancher maximale (approximative)
    surface_plancher_max = surface_emprise * nb_etages_max

    return {
        'surface_emprise': surface_emprise,
        'hauteur_moyenne': hauteur_moyenne,
        'nb_etages_estimes': nb_etages_max,
        'surface_plancher_max': surface_plancher_max
    }

# === FONCTION POUR ANALYSER LA CONFORMITÉ AVEC LE PLU ===
def analyser_conformite_plu(points_orig, points_enveloppe, hauteurs_segments, plu_regles):
    """
    Analyse la conformité de l'enveloppe réglementaire avec les règles du PLU.
    """
    resultats = {
        'conforme': True,
        'commentaires': []
    }

    # 1. Vérifier l'emprise au sol
    poly_orig = Polygon([(p[0], p[1]) for p in points_orig])
    surface_parcelle = poly_orig.area

    poly_env = Polygon([(p[0], p[1]) for p in points_enveloppe])
    emprise_sol = poly_env.area

    pourcentage_emprise = (emprise_sol / surface_parcelle) * 100

    if 'emprise_max' in plu_regles and pourcentage_emprise > plu_regles['emprise_max']:
        resultats['conforme'] = False
        resultats['commentaires'].append(
            f"Emprise au sol de {pourcentage_emprise:.1f}% supérieure au maximum autorisé ({plu_regles['emprise_max']}%)")
    else:
        resultats['commentaires'].append(
            f"Emprise au sol de {pourcentage_emprise:.1f}% conforme")

    # 2. Vérifier les hauteurs maximales
    hauteur_max_trouvee = max(hauteurs_segments)

    if 'hauteur_max' in plu_regles and hauteur_max_trouvee > plu_regles['hauteur_max']:
        resultats['conforme'] = False
        resultats['commentaires'].append(
            f"Hauteur maximale de {hauteur_max_trouvee:.1f}m supérieure au maximum autorisé ({plu_regles['hauteur_max']}m)")
    else:
        resultats['commentaires'].append(
            f"Hauteur maximale de {hauteur_max_trouvee:.1f}m conforme")

    # 3. Vérifier la surface de pleine terre minimale
    if 'pleine_terre_min' in plu_regles:
        # Calculer la surface restante (approximative) pour la pleine terre
        surface_disponible = surface_parcelle - emprise_sol
        pourcentage_pleine_terre = (surface_disponible / surface_parcelle) * 100

        if pourcentage_pleine_terre < plu_regles['pleine_terre_min']:
            resultats['conforme'] = False
            resultats['commentaires'].append(
                f"Surface de pleine terre estimée à {pourcentage_pleine_terre:.1f}% inférieure au minimum requis ({plu_regles['pleine_terre_min']}%)")
        else:
            resultats['commentaires'].append(
                f"Surface de pleine terre estimée à {pourcentage_pleine_terre:.1f}% conforme")

    return resultats

def tracer_enveloppe(analyse_id, points_orig, segments_decales, points_enveloppe=None, chemin_fichier="static/enveloppe_conforme.png"):
    """
    Trace l'enveloppe réglementaire et enregistre le résultat dans une image PNG.
    
    Arguments :
        points_orig : liste de points [(x, y, z)]
        segments_decales : liste de segments [((x1, y1), (x2, y2)), ...]
        points_enveloppe : liste optionnelle de points [(x, y)]
        chemin_fichier : chemin du fichier de sortie PNG
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Polygone original
    xs_orig, ys_orig = zip(*[(x, y) for x, y, z in points_orig])
    xs_orig, ys_orig = list(xs_orig), list(ys_orig)
    xs_orig.append(xs_orig[0])
    ys_orig.append(ys_orig[0])
    ax.plot(xs_orig, ys_orig, 'b-', linewidth=2, label="Parcelle originale")

    cmap = plt.get_cmap('hsv', len(segments_decales))

    # Segments décalés
    for i, segment in enumerate(segments_decales):
        color = cmap(i / len(segments_decales))
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        ax.plot([x1, x2], [y1, y2], '--', color=color, linewidth=2, label=f"Segment décalé {i}")

        orig_p1 = (points_orig[i][0], points_orig[i][1])
        orig_p2 = (points_orig[(i + 1) % len(points_orig)][0], points_orig[(i + 1) % len(points_orig)][1])
        ax.plot([orig_p1[0], x1], [orig_p1[1], y1], ':', color=color, alpha=0.5)
        ax.plot([orig_p2[0], x2], [orig_p2[1], y2], ':', color=color, alpha=0.5)

        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, f"{i}", color='black', fontsize=10, ha='center',
                bbox=dict(facecolor='white', alpha=0.7))

    # Enveloppe finale
    if points_enveloppe:
        x_env = [p[0] for p in points_enveloppe]
        y_env = [p[1] for p in points_enveloppe]
        x_env.append(points_enveloppe[0][0])
        y_env.append(points_enveloppe[0][1])
        ax.plot(x_env, y_env, 'r-', linewidth=3, label="Enveloppe réglementaire")

        for i, (x, y) in enumerate(points_enveloppe):
            ax.scatter(x, y, color='red', s=50)
            ax.text(x, y, f"E{i}", color='red', fontsize=10, ha='center', va='bottom')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Enveloppe réglementaire avec segments parallèles")
    ax.axis('equal')
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.tight_layout()

    # Enregistrement du fichier PNG
    
    

    analysis = Analysis.query.get(analyse_id)
    analysis_dir = "static/outputs/analyse_" + str(analysis.user_id) + "_"+ str(analyse_id) + "/"#os.path.join('static', 'analyses', str(analysis_id))
    os.makedirs(analysis_dir, exist_ok=True)

    chemin = analysis_dir + "enveloppe_conforme.png"
    plt.tight_layout()
    plt.savefig(chemin, dpi=150, format='png')
    
    if not analysis.enveloppe_conforme_img:
        analysis.enveloppe_conforme_img = {}
    # S'assurer que c'est bien un dictionnaire, au cas où la DB contient une chaîne JSON par exemple
    if isinstance(analysis.enveloppe_conforme_img, str):
        import json
        analysis.enveloppe_conforme_img = json.loads(analysis.enveloppe_conforme_img)
    analysis.enveloppe_conforme_img['enveloppe_conforme'] = chemin
    db.session.commit()

    # plt.savefig(chemin_fichier, format='png', dpi=150)
    plt.close(fig)  # Libérer les ressources

    # Optionnel : attendre un peu si tu enchaînes plusieurs fichiers
    time.sleep(0.2)

    #return chemin_fichier  # Tu peux retourner le chemin du fichier si besoin



# === FONCTION PRINCIPALE ===
def main_enveloppe_conforme(types_segments, types_murs,analysis_id):
    # Lire les coordonnées GPS depuis un fichier CSV
    transformer_fichier_parcelle(analysis_id)
    points_gps = lire_points_csv(analysis_id)#("outputs/parcelle_1.csv")

    # Nettoyage des points dupliqués
    unique_points = []
    seen = set()
    for point in points_gps:
        if point not in seen:
            seen.add(point)
            unique_points.append(point)
    points_gps = unique_points


    if not points_gps:
        print("Erreur: Aucun point valide trouvé dans le fichier CSV.")
        return

    # Convertir les coordonnées en Lambert93
    #Conversion à gerer apres en fonction des donnees reçues
    #points_lambert = convertir_en_lambert93(points_gps)
    points_lambert = points_gps

    # Générer l'enveloppe réglementaire
    segments_decales, points_enveloppe = generer_enveloppe_reglementaire(
        points_lambert, types_segments, types_murs, regles)

    # Afficher l'enveloppe 2D
    tracer_enveloppe(analysis_id, points_lambert, segments_decales, points_enveloppe)

    # Calculer les hauteurs maximales selon les règles de prospect
    # hauteurs_segments = calculer_hauteurs_prospect(
    #     points_enveloppe, segmtracer_enveloppeents_decales, types_segments, types_murs, regles)


    ######|||
    hauteurs_fixes = [regles[typ]['hauteur_max'] for typ in types_segments]
    hauteurs_segments = hauteurs_fixes
    # visualiser_enveloppe_3d(points_enveloppe, hauteurs_fixes)
    # Visualiser l'enveloppe 3D

    visualiser_3d_maillage_coherent(analysis_id, points_enveloppe, hauteurs_segments)

    # Visualiser les plans de prospect
    # visualiser_plans_prospect(points_lambert, points_enveloppe, hauteurs_segments, types_segments, types_murs)

    # Calculer la surface de plancher maximale
    resultats_surface = calculer_surface_plancher_max(points_enveloppe, hauteurs_segments)
    # print("\n=== RÉSULTATS ===")
    print(f"Surface d'emprise au sol: {resultats_surface['surface_emprise']:.1f} m²")
    # print(f"Hauteur moyenne: {resultats_surface['hauteur_moyenne']:.1f} m")
    # print(f"Nombre d'étages estimé: {resultats_surface['nb_etages_estimes']}")
    # print(f"Surface de plancher maximale estimée: {resultats_surface['surface_plancher_max']:.1f} m²")

    # Analyser la conformité avec le PLU
    #
    # resultats_conformite = analyser_conformite_plu(
        # points_lambert, points_enveloppe, hauteurs_segments, plu_regles_zone_UAs)

    # print("\n=== ANALYSE DE CONFORMITÉ PLU ===")
    # print(f"Conforme: {'Oui' if resultats_conformite['conforme'] else 'Non'}")
    # for commentaire in resultats_conformite['commentaires']:
    #     print(f"- {commentaire}")

    # # Exporter les résultats
    # exporter = input("\nSouhaitez-vous exporter les résultats dans un fichier CSV? (o/n): ")
    # if exporter.lower() == 'o':
    #     # exporter_resultats(points_enveloppe, hauteurs_segmentss)
    exporter_points_csv(analysis_id, points_enveloppe, hauteurs_segments)
    print("Fin Analyse")
