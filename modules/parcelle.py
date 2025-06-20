import ifcopenshell
import ifcopenshell.geom
import numpy as np
import pandas as pd
import os
from collections import defaultdict, Counter
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from models import Analysis,db


def extraire_points_emprise(ifc_path, analysis_id): #csv_output_path="outputs/coords_dessin_polygone.csv"):
    """
    Extrait les coordonnées des points de contour de toutes les parcelles (EMPRISE)
    avec une approche robuste multi-méthodes.
    """
    # Créer le dossier de sortie s'il n'existe pas
    #os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    # Ouvrir le fichier IFC
    ifc_file = ifcopenshell.open(ifc_path)

    # Liste pour stocker les informations sur les parcelles
    parcelles_data = []

    # Trouver tous les éléments avec la catégorie "EMPRISE"
    elements_emprise = trouver_elements_emprise(ifc_file)

    if not elements_emprise:
        print("❌ Aucun élément avec catégorie 'EMPRISE' n'a été trouvé.")
        return None

    print(f"🔍 {len(elements_emprise)} parcelles (EMPRISE) trouvées dans le fichier IFC.")

    # Configurer les paramètres de géométrie
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    # Pour chaque élément EMPRISE, extraire les points du contour
    for idx, element in enumerate(elements_emprise):
        id_parcelle = idx + 1

        try:
            print(f"🔄 Traitement de la parcelle {id_parcelle}...")
            
            # Créer la forme géométrique
            shape = ifcopenshell.geom.create_shape(settings, element)

            # Extraire les sommets et les faces
            verts = shape.geometry.verts
            faces = shape.geometry.faces

            print(f"   📊 Géométrie: {len(verts)//3} vertices, {len(faces)//3} faces")

            # Convertir les sommets en tableau numpy
            vertices = np.array(verts).reshape(-1, 3)

            # Essayer plusieurs méthodes dans l'ordre
            contour_points = []
            
            # Méthode 1: Extraction directe des vertices uniques
            print("   🔍 Méthode 1: Extraction directe des vertices...")
            contour_points = methode_vertices_directs(vertices)
            
            if not contour_points:
                print("   🔍 Méthode 2: Analyse des arêtes...")
                contour_points = methode_analyse_aretes(vertices, faces)
            
            if not contour_points:
                print("   🔍 Méthode 3: Projection 2D avec tri radial...")
                contour_points = methode_tri_radial(vertices)
            
            if not contour_points:
                print("   🔍 Méthode 4: Boundaries géométriques...")
                contour_points = methode_boundaries_geometriques(vertices, faces)

            # Ajouter les points à la liste des données
            if contour_points:
                for point in contour_points:
                    parcelles_data.append({
                        "id_parcelle": id_parcelle,
                        "X": point[1],  # Y dans IFC devient X dans CSV
                        "Y": point[0]   # X dans IFC devient Y dans CSV
                    })
                print(f"✅ Parcelle {id_parcelle}: {len(contour_points)} points extraits")
            else:
                print(f"❌ Parcelle {id_parcelle}: Aucun point extrait avec toutes les méthodes")

        except Exception as e:
            print(f"⚠️ Erreur lors de l'extraction de la parcelle {id_parcelle}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Créer un DataFrame à partir des données collectées
    if parcelles_data:
        df_points = pd.DataFrame(parcelles_data)
        df_points["Z"] = 0.0 

        #df_points.to_csv(csv_output_path, index=False, float_format="%.1f")
        #print(f"✅ Points exportés vers : {csv_output_path}")
        points = df_points.to_dict(orient='records')  # Convertit en [{id_parcelle:1, X:..., Y:..., Z:...}]
    
        analysis = Analysis.query.get(analysis_id)
        analysis.parcelle_data = {"version": "1.0", "points": points}
        db.session.commit()
        return df_points
    else:
        print("❌ Aucun point n'a pu être extrait.")
        return None


def methode_vertices_directs(vertices):
    """
    Méthode 1: Extraction directe des vertices uniques en 2D
    """
    try:
        # Convertir en 2D et supprimer les doublons
        points_2d = vertices[:, :2]
        
        # Arrondir pour éviter les problèmes de précision flottante
        points_rounded = np.round(points_2d, decimals=1)
        
        # Supprimer les doublons
        unique_points = np.unique(points_rounded, axis=0)
        
        print(f"      → {len(unique_points)} points uniques trouvés")
        
        if len(unique_points) >= 3:
            # Trier les points dans l'ordre du contour
            contour_points = trier_points_contour(unique_points)
            return [(point[0], point[1]) for point in contour_points]
        
        return []
        
    except Exception as e:
        print(f"      ❌ Erreur méthode vertices directs: {str(e)}")
        return []


def methode_analyse_aretes(vertices, faces):
    """
    Méthode 2: Analyse des arêtes pour trouver les boundaries
    """
    try:
        if len(faces) == 0:
            return []
            
        # Créer un dictionnaire pour compter les arêtes
        edge_count = Counter()
        
        # Parcourir toutes les faces
        for i in range(0, len(faces), 3):
            if i + 2 < len(faces):
                v1, v2, v3 = faces[i], faces[i+1], faces[i+2]
                
                # Créer les arêtes (triées pour éviter les doublons)
                edges = [
                    tuple(sorted([v1, v2])),
                    tuple(sorted([v2, v3])),
                    tuple(sorted([v3, v1]))
                ]
                
                for edge in edges:
                    edge_count[edge] += 1
        
        # Les arêtes de bordure apparaissent une seule fois
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        print(f"      → {len(boundary_edges)} arêtes de bordure trouvées")
        
        if boundary_edges:
            # Extraire les points uniques des arêtes de bordure
            boundary_vertices = set()
            for edge in boundary_edges:
                boundary_vertices.update(edge)
            
            # Convertir en coordonnées 2D
            boundary_points = []
            for vertex_idx in boundary_vertices:
                if vertex_idx < len(vertices):
                    point = vertices[vertex_idx][:2]
                    boundary_points.append((point[0], point[1]))
            
            return boundary_points
        
        return []
        
    except Exception as e:
        print(f"      ❌ Erreur méthode analyse arêtes: {str(e)}")
        return []


def methode_tri_radial(vertices):
    """
    Méthode 3: Tri radial des points autour du centroïde
    """
    try:
        points_2d = vertices[:, :2]
        unique_points = np.unique(np.round(points_2d, decimals=1), axis=0)
        
        if len(unique_points) < 3:
            return [(point[0], point[1]) for point in unique_points]
        
        # Calculer le centroïde
        centroid = np.mean(unique_points, axis=0)
        
        # Calculer l'angle de chaque point par rapport au centroïde
        angles = []
        for point in unique_points:
            angle = math.atan2(point[1] - centroid[1], point[0] - centroid[0])
            angles.append((angle, point))
        
        # Trier par angle
        angles.sort(key=lambda x: x[0])
        
        # Extraire les points triés
        sorted_points = [point for angle, point in angles]
        
        print(f"      → {len(sorted_points)} points triés radialement")
        
        return [(point[0], point[1]) for point in sorted_points]
        
    except Exception as e:
        print(f"      ❌ Erreur méthode tri radial: {str(e)}")
        return []


def methode_boundaries_geometriques(vertices, faces):
    """
    Méthode 4: Extraction des boundaries géométriques
    """
    try:
        points_2d = vertices[:, :2]
        unique_points = np.unique(np.round(points_2d, decimals=1), axis=0)
        
        if len(unique_points) < 3:
            return [(point[0], point[1]) for point in unique_points]
        
        # Trouver les points extrêmes
        min_x_idx = np.argmin(unique_points[:, 0])
        max_x_idx = np.argmax(unique_points[:, 0])
        min_y_idx = np.argmin(unique_points[:, 1])
        max_y_idx = np.argmax(unique_points[:, 1])
        
        extreme_indices = {min_x_idx, max_x_idx, min_y_idx, max_y_idx}
        
        # Commencer par tous les points et les trier
        all_points = [(point[0], point[1]) for point in unique_points]
        
        # Si on a des points extrêmes, les prioriser
        if len(extreme_indices) >= 3:
            extreme_points = [unique_points[i] for i in extreme_indices]
            sorted_extreme = trier_points_contour(np.array(extreme_points))
            
            # Ajouter les autres points
            remaining_points = []
            for i, point in enumerate(unique_points):
                if i not in extreme_indices:
                    remaining_points.append(point)
            
            if remaining_points:
                all_combined = np.vstack([sorted_extreme, remaining_points])
                final_sorted = trier_points_contour(all_combined)
                return [(point[0], point[1]) for point in final_sorted]
            else:
                return [(point[0], point[1]) for point in sorted_extreme]
        
        return all_points
        
    except Exception as e:
        print(f"      ❌ Erreur méthode boundaries géométriques: {str(e)}")
        return []


def trier_points_contour(points):
    """
    Trie les points pour former un contour cohérent
    """
    if len(points) < 3:
        return points
    
    # Calculer le centroïde
    centroid = np.mean(points, axis=0)
    
    # Calculer les angles et trier
    angles_points = []
    for point in points:
        angle = math.atan2(point[1] - centroid[1], point[0] - centroid[0])
        angles_points.append((angle, point))
    
    # Trier par angle
    angles_points.sort(key=lambda x: x[0])
    
    # Retourner les points triés
    return np.array([point for angle, point in angles_points])


def trouver_elements_emprise(ifc_file):
    """
    Trouve tous les éléments ayant une propriété Category avec la valeur "EMPRISE"
    """
    elements_emprise = []

    for pset in ifc_file.by_type("IFCPROPERTYSET"):
        for prop in pset.HasProperties:
            if (prop.is_a("IFCPROPERTYSINGLEVALUE") and
                prop.Name == "Category" and
                prop.NominalValue and
                prop.NominalValue.wrappedValue == "EMPRISE"):

                for rel in ifc_file.by_type("IFCRELDEFINESBYPROPERTIES"):
                    if rel.RelatingPropertyDefinition == pset:
                        elements_emprise.extend(rel.RelatedObjects)

    return elements_emprise








#########################################################################################
def visualiser_parcelle(analysis_id, output_dir="static/parcelle"): #(csv_path="outputs/coords_dessin_polygone.csv", output_dir="static/parcelle"):
    """
    Génère une visualisation PNG pour chaque parcelle définie dans le fichier CSV
    et retourne les informations des segments.

    Args:
        csv_path (str): Chemin vers le fichier CSV contenant les coordonnées des parcelles
        output_dir (str): Dossier de sortie pour les images PNG

    Returns:
        dict: Dictionnaire contenant les segments_info pour chaque parcelle
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Charger les données du CSV
    #df = pd.read_csv(csv_path)
     # Récupérer l'analyse depuis la base
    analysis = Analysis.query.get(analysis_id)
    if not analysis or not analysis.parcelle_data:
        print("❌ Analyse ou données de parcelle introuvables")
        return None

    # Convertir les données JSON en DataFrame
    try:
        points_data = analysis.parcelle_data.get('points', [])
        df = pd.DataFrame(points_data)
        
        # Renommer les colonnes pour compatibilité avec le code existant
        df = df.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'})
        
        # Vérification des données
        if df.empty or 'X' not in df.columns or 'Y' not in df.columns:
            print("❌ Structure de données invalide")
            return None

    except Exception as e:
        print(f"❌ Erreur de conversion des données: {str(e)}")
        return None

    # Vérifier que le CSV contient les colonnes attendues
    if not all(col in df.columns for col in ["id_parcelle", "X", "Y"]):
        print("❌ Le CSV doit contenir les colonnes: id_parcelle, X, Y")
        return None

    # Identifier les parcelles uniques
    parcelles_ids = df["id_parcelle"].unique()
    print(f"🔍 {len(parcelles_ids)} parcelles trouvées dans le CSV.")

    # Couleurs pour les segments
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Dictionnaire pour stocker les segments_info de toutes les parcelles
    all_segments_info = {}

    # Pour chaque parcelle, générer une visualisation
    for parcelle_id in parcelles_ids:
        # Filtrer les points pour cette parcelle
        points_parcelle = df[df["id_parcelle"] == parcelle_id]

        # Extraire les coordonnées X et Y
        x_coords = points_parcelle["X"].values
        y_coords = points_parcelle["Y"].values

        # Créer un tableau de points (vertices)
        vertices = np.column_stack([x_coords, y_coords])
        n_segments = len(vertices)

        # Vérifier qu'il y a suffisamment de points
        if n_segments < 3:
            print(f"⚠️ Pas assez de points pour la parcelle {parcelle_id}, visualisation ignorée.")
            continue

        # Créer la figure et les axes
        fig, ax = plt.subplots(figsize=(10, 10))

        # Liste pour stocker les informations des segments de cette parcelle
        segments_info = []

        # Dessiner les segments avec des couleurs différentes et numérotés
        for i in range(n_segments):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n_segments]
            
            # Choisir une couleur pour ce segment
            color = colors[i % len(colors)]

            # Dessiner le segment
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=3)
            
            # Calculer le point milieu pour placer le numéro du segment
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            
            # Ajouter le numéro du segment
            ax.text(mid_x, mid_y, str(i), fontsize=16, 
                   bbox=dict(facecolor='white', edgecolor='black'))

            # Ajouter les informations du segment
            segments_info.append({
                "id": i,
                "start": {"x": round(float(p1[0]), 2), "y": round(float(p1[1]), 2)},
                "end": {"x": round(float(p2[0]), 2), "y": round(float(p2[1]), 2)},
            })

        # Stocker les segments_info pour cette parcelle
        all_segments_info[parcelle_id] = segments_info

        # Calculer le centre de la parcelle
        center_x = np.mean(vertices[:, 0])
        center_y = np.mean(vertices[:, 1])

        # Ajouter l'ID de la parcelle au centre
        ax.text(center_x, center_y, f"Parcelle {parcelle_id}", fontsize=14, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))

        # Définir les limites des axes
        margin = 0.1 * max(np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1]))
        ax.set_xlim(np.min(vertices[:, 0]) - margin, np.max(vertices[:, 0]) + margin)
        ax.set_ylim(np.min(vertices[:, 1]) - margin, np.max(vertices[:, 1]) + margin)

        # Ajouter le titre et les étiquettes d'axes
        # ax.set_title(f"Parcelle {parcelle_id}", fontsize=16)
        ax.set_title(f"Parcelle", fontsize=16)
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Enregistrer l'image
        output_path = os.path.join(output_dir, f"parcelle_{parcelle_id}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"✅ Visualisation générée pour la parcelle {parcelle_id}: {output_path}")
        print(f"   📊 {len(segments_info)} segments extraits pour cette parcelle")

    print(f"✅ Toutes les visualisations ont été générées dans: {output_dir}")
    
    # Retourner les informations des segments
    return all_segments_info







# # Exemple d'utilisation
# if __name__ == "__main__":
#     # Remplacez par le chemin de votre fichier IFC
#     ifc_path = "AAP_BIM.ifc"
#     csv_output = "outputs/coords_parcelles.csv"
    
#     df_result = coords_dessin_polygone(ifc_path, csv_output)
    
#     if df_result is not None:
#         print(f"Extraction terminée. {len(df_result)} points extraits.")
#         visualiser_parcelles_standalone(csv_path)
#     else:
#         print("Échec de l'extraction.")


        