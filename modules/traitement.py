import ifcopenshell
import ifcopenshell.geom
import numpy as np
import pandas as pd
import csv
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.polygon import orient
import math
import datetime

from models import Analysis,db

def extraire_points_emprise(ifc_path, analysis_id): #csv_output_path="outputs/coords_dessin_polygone.csv"):
    ifc_file = ifcopenshell.open(ifc_path)

    def trouver_element_emprise(ifc_file):
        for pset in ifc_file.by_type("IFCPROPERTYSET"):
            for prop in pset.HasProperties:
                if (
                    prop.is_a("IFCPROPERTYSINGLEVALUE")
                    and prop.Name == "Category"
                    and prop.NominalValue.wrappedValue == "EMPRISE"
                ):
                    for rel in ifc_file.by_type("IFCRELDEFINESBYPROPERTIES"):
                        if rel.RelatingPropertyDefinition == pset:
                            return rel.RelatedObjects[0]
        return None

    element = trouver_element_emprise(ifc_file)
    if not element:
        print("Élément avec catégorie 'EMPRISE' non trouvé.")
        return None

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    shape = ifcopenshell.geom.create_shape(settings, element)
    verts = shape.geometry.verts
    faces = shape.geometry.faces

    vertices = np.array(verts).reshape(-1, 3)
    contour_points = []
    processed_edges = set()

    for i in range(0, len(faces), 3):
        v1, v2, v3 = faces[i], faces[i+1], faces[i+2]
        for edge in [(min(v1, v2), max(v1, v2)), (min(v2, v3), max(v2, v3)), (min(v3, v1), max(v3, v1))]:
            if edge not in processed_edges:
                processed_edges.add(edge)
                p1, p2 = vertices[edge[0]], vertices[edge[1]]
                contour_points.append((p1[0], p1[1]))
                contour_points.append((p2[0], p2[1]))

    unique_points = list(set(contour_points))
    unique_points = [(y, x) for (x, y) in unique_points]  # inversion pour X=Y IFC → X=Y CSV

    df_points = pd.DataFrame(unique_points, columns=["X", "Y"])
    df_points["Z"] = 0.0  # Ajout de la colonne Z avec valeur 0.0

    #df_points.to_csv(csv_output_path, index=False, float_format="%.1f")
    ##DB use
    points = df_points.to_dict(orient='records')  # Convertit en [{id_parcelle:1, X:..., Y:..., Z:...}]
    
    analysis = Analysis.query.get(analysis_id)
    analysis.parcelle_data = {"version": "1.0", "points": points}
    db.session.commit()
    #print(f"✅ Points exportés vers : {csv_output_path}")
    print("DEBUT")
    print(df_points)
    print("FIN")
    return df_points

# def extraire_points_emprise(ifc_path, analysis_id):
#     ifc_file = ifcopenshell.open(ifc_path)
    
#     # Trouver l'élément EMPRISE (version optimisée)
#     element = next(
#         (e for e in ifc_file.by_type("IFCBUILDINGELEMENTPART") 
#          if any(p.Name == "Category" and p.NominalValue.wrappedValue == "EMPRISE"
#                for p in e.IsDefinedBy[0].RelatingPropertyDefinition.HasProperties)),
#         None
#     )
    
#     if not element:
#         print("❌ Aucun élément EMPRISE trouvé")
#         return None

#     # Extraction des points
#     settings = ifcopenshell.geom.settings()
#     settings.set(settings.USE_WORLD_COORDS, True)
#     shape = ifcopenshell.geom.create_shape(settings, element)
#     verts = np.array(shape.geometry.verts).reshape(-1, 3)
    
#     # Formatage des points (avec inversion X/Y)
#     points_data = [
#         {"x": float(v[1]), "y": float(v[0]), "z": 0.0}
#         for v in verts
#     ]

#     # Mise à jour de l'analyse
#     analysis = Analysis.query.get(analysis_id)
#     if not analysis:
#         print(f"❌ Analyse {analysis_id} introuvable")
#         return None
        
#     analysis.parcelle_data = {
#         "points": points_data,
#         "metadata": {
#             "source": "IFC",
#             "extracted_at": datetime.utcnow().isoformat(),
#             "point_count": len(points_data)
#         }
#     }
    
#     # Création du DataFrame pour compatibilité ascendante
#     df = pd.DataFrame([
#         {"X": p["x"], "Y": p["y"], "Z": p["z"]}
#         for p in points_data
#     ])
    
#     db.session.commit()
#     return df


# import ifcopenshell
# import numpy as np
# import pandas as pd
# from scipy.spatial import ConvexHull
# import matplotlib.pyplot as plt

# def extraire_points_empriseee(ifc_path):
#     """Extrait les coordonnées de toutes les parcelles 'EMPRISE' avec contours fermés."""
#     ifc_file = ifcopenshell.open(ifc_path)
    
#     # Étape 1: Trouver tous les éléments EMPRISE
#     elements_emprise = []
#     for pset in ifc_file.by_type("IFCPROPERTYSET"):
#         for prop in pset.HasProperties:
#             if (prop.is_a("IFCPROPERTYSINGLEVALUE") and 
#                 prop.Name == "Category" and 
#                 prop.NominalValue.wrappedValue == "EMPRISE"):
#                 for rel in ifc_file.by_type("IFCRELDEFINESBYPROPERTIES"):
#                     if rel.RelatingPropertyDefinition == pset:
#                         elements_emprise.extend(rel.RelatedObjects)
    
#     if not elements_emprise:
#         return None
    
#     settings = ifcopenshell.geom.settings()
#     settings.set(settings.USE_WORLD_COORDS, True)
    
#     all_contours = []
    
#     for idx, element in enumerate(elements_emprise, start=1):
#         try:
#             shape = ifcopenshell.geom.create_shape(settings, element)
#             verts = shape.geometry.verts
#             faces = shape.geometry.faces
            
#             # Extraction des points uniques
#             vertices = np.array(verts).reshape(-1, 3)
#             unique_points = np.unique(vertices[:, :2], axis=0)  # Prendre X,Y et supprimer doublons
            
#             # Création du contour fermé
#             if len(unique_points) >= 3:
#                 hull = ConvexHull(unique_points)
#                 contour = unique_points[hull.vertices]
#                 contour = np.vstack([contour, contour[0]])  # Fermer le polygone
                
#                 # Stockage avec l'ID de parcelle
#                 for point in contour:
#                     all_contours.append({
#                         'id_parcelle': f"Parcelle_{idx}",
#                         'X': point[0],
#                         'Y': point[1]
#                     })
                
#         except Exception as e:
#             print(f"Erreur parcelle {idx}: {str(e)}")
#             continue

#     return pd.DataFrame(all_contours) if all_contours else None
###############################################################################


###############################################################################
def enregistrer_vertices_csv(vertices, fichier_csv="static/_coords_converted.csv"):
    """
    Enregistre une liste ou un tableau numpy de points (x, y) dans un fichier CSV au format :
    x;y
    avec 4 décimales, et le premier point répété à la fin.
    
    Args:
        vertices (list or np.ndarray): Liste ou array de coordonnées (en Lambert93)
        fichier_csv (str): Nom du fichier de sortie
    """
    if isinstance(vertices, np.ndarray):
        vertices = vertices.tolist()  # Conversion en liste si besoin

    if len(vertices) == 0:
        print("Aucun sommet à enregistrer.")
        return

    # Répéter le premier point à la fin pour fermer le polygone
    closed_vertices = vertices[:]
    if closed_vertices[0] != closed_vertices[-1]:
        closed_vertices.append(closed_vertices[0])

    with open(fichier_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        for x, y in closed_vertices:
            writer.writerow([f"{x:.4f}", f"{y:.4f}"])

    print(f"{len(closed_vertices)} points enregistrés dans {fichier_csv}")
