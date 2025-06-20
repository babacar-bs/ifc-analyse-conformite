import csv
import math
import ifcopenshell
from models import *
import pandas as pd


# ✅ Métadonnées
def create_owner_history(ifc_file):
    person = ifc_file.create_entity("IfcPerson", FamilyName="SEVE-UP")
    org = ifc_file.create_entity("IfcOrganization", Name="SEVE-UP")
    person_and_org = ifc_file.create_entity("IfcPersonAndOrganization", ThePerson=person, TheOrganization=org)
    app = ifc_file.create_entity(
        "IfcApplication",
        ApplicationDeveloper=org,
        Version="1.0",
        ApplicationFullName="Analyse-Conformite",
        ApplicationIdentifier="APP001"
    )
    return ifc_file.create_entity("IfcOwnerHistory", OwningUser=person_and_org, OwningApplication=app)

# ✅ Couleur
def add_global_style(model, shape, rgb=(0.0, 0.5, 1.0)):
    color = model.create_entity("IfcColourRgb", Name="Color", Red=rgb[0], Green=rgb[1], Blue=rgb[2])
    surface_style = model.create_entity(
        "IfcSurfaceStyle",
        Name="SurfaceStyle",
        Side="BOTH",
        Styles=[model.create_entity("IfcSurfaceStyleRendering", SurfaceColour=color)]
    )
    style_assignment = model.create_entity("IfcPresentationStyleAssignment", Styles=[surface_style])
    model.create_entity("IfcStyledItem", Item=shape, Styles=[style_assignment])

# ✅ Altitude de référence depuis un IFC
def get_building_storey_info(ifc_path):
    ifc = ifcopenshell.open(ifc_path)
    storeys = ifc.by_type("IfcBuildingStorey")
    if not storeys:
        print("❌ Aucun IfcBuildingStorey trouvé.")
        return 0.0
    return float(storeys[0].Elevation or 0.0)

# ✅ Création IFC robuste avec sommets inclinés partagés

def create_point(ifc_file, cache, x, y, z, precision=4):
    key = (round(x, precision), round(y, precision), round(z, precision))
    if key not in cache:
        cache[key] = ifc_file.create_entity("IfcCartesianPoint", Coordinates=key)
    return cache[key]

# def create_ifc_from_csv(csv_path, output_path, altitude=0.0):
#     ifc_file = ifcopenshell.file(schema="IFC2X3")
#     owner_history = create_owner_history(ifc_file)

#     units = ifc_file.create_entity("IfcUnitAssignment", Units=[
#         ifc_file.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
#     ])

#     context = ifc_file.create_entity("IfcGeometricRepresentationContext",
#         ContextIdentifier="Plan",
#         ContextType="Model",
#         CoordinateSpaceDimension=3,
#         Precision=1e-05,
#         WorldCoordinateSystem=ifc_file.create_entity("IfcAxis2Placement3D",
#             Location=ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
#         )
#     )

#     project = ifc_file.create_entity("IfcProject",
#         GlobalId=ifcopenshell.guid.new(),
#         Name="Projet",
#         OwnerHistory=owner_history,
#         RepresentationContexts=[context],
#         UnitsInContext=units
#     )

#     site_placement = ifc_file.create_entity("IfcLocalPlacement",
#         RelativePlacement=ifc_file.create_entity("IfcAxis2Placement3D",
#             Location=ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
#         )
#     )

#     site = ifc_file.create_entity("IfcSite",
#         GlobalId=ifcopenshell.guid.new(),
#         OwnerHistory=owner_history,
#         Name="Site",
#         ObjectPlacement=site_placement,
#         CompositionType="ELEMENT"
#     )

#     building = ifc_file.create_entity("IfcBuilding",
#         GlobalId=ifcopenshell.guid.new(),
#         OwnerHistory=owner_history,
#         Name="Bâtiment",
#         ObjectPlacement=site_placement,
#         CompositionType="ELEMENT"
#     )

#     storey = ifc_file.create_entity("IfcBuildingStorey",
#         GlobalId=ifcopenshell.guid.new(),
#         OwnerHistory=owner_history,
#         Name="Niveau",
#         ObjectPlacement=site_placement,
#         CompositionType="ELEMENT",
#         Elevation=altitude
#     )

#     ifc_file.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history,
#                            RelatingObject=project, RelatedObjects=[site])
#     ifc_file.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history,
#                            RelatingObject=site, RelatedObjects=[building])
#     ifc_file.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history,
#                            RelatingObject=building, RelatedObjects=[storey])

#     point_cache = {}
#     inclined_cache = {}  # Cache des sommets inclinés partagés
#     all_coords = []
#     with open(csv_path, newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             all_coords.append((float(row['X1']), float(row['Y1'])))
#             all_coords.append((float(row['X2']), float(row['Y2'])))
#     cx = sum(x for x, y in all_coords) / len(all_coords)
#     cy = sum(y for x, y in all_coords) / len(all_coords)

#     with open(csv_path, newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             x1 = float(row['X1'])
#             y1 = float(row['Y1'])
#             x2 = float(row['X2'])
#             y2 = float(row['Y2'])
#             h = float(row['Hauteur'])
#             retrait = float(row.get('Retrait', 0))
#             type_facade = row.get('Type', 'NA')

#             dx = x2 - x1
#             dy = y2 - y1
#             length = math.hypot(dx, dy)
#             dx_norm = dx / length
#             dy_norm = dy / length

#             nx = -dy_norm
#             ny = dx_norm
#             xm = (x1 + x2) / 2
#             ym = (y1 + y2) / 2
#             vx = cx - xm
#             vy = cy - ym
#             if (nx * vx + ny * vy) < 0:
#                 nx *= -1
#                 ny *= -1

#             if type_facade == 'VO' and retrait >= 2:
#                 h1 = h - retrait
#                 p1 = create_point(ifc_file, point_cache, x1, y1, altitude)
#                 p2 = create_point(ifc_file, point_cache, x2, y2, altitude)
#                 p3 = create_point(ifc_file, point_cache, x2, y2, altitude + h1)
#                 p4 = create_point(ifc_file, point_cache, x1, y1, altitude + h1)

#                 # Sommets inclinés partagés
#                 key5 = (round(x2, 3), round(y2, 3))
#                 key6 = (round(x1, 3), round(y1, 3))
#                 if key5 not in inclined_cache:
#                     inclined_cache[key5] = create_point(ifc_file, point_cache, x2 + nx * retrait, y2 + ny * retrait, altitude + h)
#                 if key6 not in inclined_cache:
#                     inclined_cache[key6] = create_point(ifc_file, point_cache, x1 + nx * retrait, y1 + ny * retrait, altitude + h)
#                 p5 = inclined_cache[key5]
#                 p6 = inclined_cache[key6]

#                 loop1 = ifc_file.create_entity("IfcPolyLoop", Polygon=[p1, p2, p3, p4, p1])
#                 loop2 = ifc_file.create_entity("IfcPolyLoop", Polygon=[p4, p3, p5, p6, p4])
#                 face1 = ifc_file.create_entity("IfcFace", Bounds=[ifc_file.create_entity("IfcFaceOuterBound", Bound=loop1, Orientation=True)])
#                 face2 = ifc_file.create_entity("IfcFace", Bounds=[ifc_file.create_entity("IfcFaceOuterBound", Bound=loop2, Orientation=True)])
#                 face_set = ifc_file.create_entity("IfcConnectedFaceSet", CfsFaces=[face1, face2])
#                 brep = ifc_file.create_entity("IfcFacetedBrep", Outer=face_set)
#             else:
#                 p1 = create_point(ifc_file, point_cache, x1, y1, altitude)
#                 p2 = create_point(ifc_file, point_cache, x2, y2, altitude)
#                 p3 = create_point(ifc_file, point_cache, x2, y2, altitude + h)
#                 p4 = create_point(ifc_file, point_cache, x1, y1, altitude + h)
#                 loop = ifc_file.create_entity("IfcPolyLoop", Polygon=[p1, p2, p3, p4, p1])
#                 bound = ifc_file.create_entity("IfcFaceOuterBound", Bound=loop, Orientation=True)
#                 face = ifc_file.create_entity("IfcFace", Bounds=[bound])
#                 face_set = ifc_file.create_entity("IfcConnectedFaceSet", CfsFaces=[face])
#                 brep = ifc_file.create_entity("IfcFacetedBrep", Outer=face_set)

#             shape_rep = ifc_file.create_entity("IfcShapeRepresentation",
#                 ContextOfItems=context,
#                 RepresentationIdentifier="Body",
#                 RepresentationType="Brep",
#                 Items=[brep]
#             )

#             add_global_style(ifc_file, brep)

#             rep = ifc_file.create_entity("IfcProductDefinitionShape", Representations=[shape_rep])

#             wall = ifc_file.create_entity("IfcBuildingElementProxy",
#                 GlobalId=ifcopenshell.guid.new(),
#                 OwnerHistory=owner_history,
#                 Name=f"Mur_{row['Segment']}",
#                 Representation=rep,
#                 ObjectPlacement=site_placement
#             )

#             ifc_file.create_entity("IfcRelContainedInSpatialStructure",
#                 GlobalId=ifcopenshell.guid.new(),
#                 OwnerHistory=owner_history,
#                 RelatingStructure=storey,
#                 RelatedElements=[wall]
#             )

#     ifc_file.write(output_path)
#     print(f"\u2705 Fichier IFC créé avec succès : {output_path}")

def create_ifc_from_csv(analysis_id, output_path, altitude=0.0):
    analysis = Analysis.query.get(analysis_id)
    if not analysis or not analysis.enveloppe_finale:
        print(f"❌ Analyse {analysis_id} introuvable ou enveloppe_finale vide")
        return False

    data = analysis.enveloppe_finale

    ifc_file = ifcopenshell.file(schema="IFC2X3")
    owner_history = create_owner_history(ifc_file)

    units = ifc_file.create_entity("IfcUnitAssignment", Units=[
        ifc_file.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
    ])

    context = ifc_file.create_entity("IfcGeometricRepresentationContext",
        ContextIdentifier="Plan",
        ContextType="Model",
        CoordinateSpaceDimension=3,
        Precision=1e-05,
        WorldCoordinateSystem=ifc_file.create_entity("IfcAxis2Placement3D",
            Location=ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
        )
    )

    project = ifc_file.create_entity("IfcProject",
        GlobalId=ifcopenshell.guid.new(),
        Name="Projet",
        OwnerHistory=owner_history,
        RepresentationContexts=[context],
        UnitsInContext=units
    )

    site_placement = ifc_file.create_entity("IfcLocalPlacement",
        RelativePlacement=ifc_file.create_entity("IfcAxis2Placement3D",
            Location=ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
        )
    )

    site = ifc_file.create_entity("IfcSite",
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=owner_history,
        Name="Site",
        ObjectPlacement=site_placement,
        CompositionType="ELEMENT"
    )

    building = ifc_file.create_entity("IfcBuilding",
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=owner_history,
        Name="Bâtiment",
        ObjectPlacement=site_placement,
        CompositionType="ELEMENT"
    )

    storey = ifc_file.create_entity("IfcBuildingStorey",
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=owner_history,
        Name="Niveau",
        ObjectPlacement=site_placement,
        CompositionType="ELEMENT",
        Elevation=altitude
    )

    # Relier la hiérarchie
    ifc_file.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history,
                           RelatingObject=project, RelatedObjects=[site])
    ifc_file.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history,
                           RelatingObject=site, RelatedObjects=[building])
    ifc_file.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history,
                           RelatingObject=building, RelatedObjects=[storey])

    point_cache = {}
    inclined_cache = {}

    all_coords = [(float(row['X1']), float(row['Y1'])) for row in data] + \
                 [(float(row['X2']), float(row['Y2'])) for row in data]
    cx = sum(x for x, y in all_coords) / len(all_coords)
    cy = sum(y for x, y in all_coords) / len(all_coords)

    for row in data:
        x1 = float(row['X1'])
        y1 = float(row['Y1'])
        x2 = float(row['X2'])
        y2 = float(row['Y2'])
        h = float(row['hauteur'])
        retrait = float(row.get('Retrait', 0))
        type_facade = row.get('Type', 'NA')

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        dx_norm = dx / length
        dy_norm = dy / length

        nx = -dy_norm
        ny = dx_norm
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        vx = cx - xm
        vy = cy - ym
        if (nx * vx + ny * vy) < 0:
            nx *= -1
            ny *= -1

        if type_facade == 'VO' and retrait >= 2:
            h1 = h - retrait
            p1 = create_point(ifc_file, point_cache, x1, y1, altitude)
            p2 = create_point(ifc_file, point_cache, x2, y2, altitude)
            p3 = create_point(ifc_file, point_cache, x2, y2, altitude + h1)
            p4 = create_point(ifc_file, point_cache, x1, y1, altitude + h1)

            key5 = (round(x2, 3), round(y2, 3))
            key6 = (round(x1, 3), round(y1, 3))
            if key5 not in inclined_cache:
                inclined_cache[key5] = create_point(ifc_file, point_cache, x2 + nx * retrait, y2 + ny * retrait, altitude + h)
            if key6 not in inclined_cache:
                inclined_cache[key6] = create_point(ifc_file, point_cache, x1 + nx * retrait, y1 + ny * retrait, altitude + h)
            p5 = inclined_cache[key5]
            p6 = inclined_cache[key6]

            loop1 = ifc_file.create_entity("IfcPolyLoop", Polygon=[p1, p2, p3, p4, p1])
            loop2 = ifc_file.create_entity("IfcPolyLoop", Polygon=[p4, p3, p5, p6, p4])
            face1 = ifc_file.create_entity("IfcFace", Bounds=[ifc_file.create_entity("IfcFaceOuterBound", Bound=loop1, Orientation=True)])
            face2 = ifc_file.create_entity("IfcFace", Bounds=[ifc_file.create_entity("IfcFaceOuterBound", Bound=loop2, Orientation=True)])
            face_set = ifc_file.create_entity("IfcConnectedFaceSet", CfsFaces=[face1, face2])
            brep = ifc_file.create_entity("IfcFacetedBrep", Outer=face_set)
        else:
            p1 = create_point(ifc_file, point_cache, x1, y1, altitude)
            p2 = create_point(ifc_file, point_cache, x2, y2, altitude)
            p3 = create_point(ifc_file, point_cache, x2, y2, altitude + h)
            p4 = create_point(ifc_file, point_cache, x1, y1, altitude + h)
            loop = ifc_file.create_entity("IfcPolyLoop", Polygon=[p1, p2, p3, p4, p1])
            bound = ifc_file.create_entity("IfcFaceOuterBound", Bound=loop, Orientation=True)
            face = ifc_file.create_entity("IfcFace", Bounds=[bound])
            face_set = ifc_file.create_entity("IfcConnectedFaceSet", CfsFaces=[face])
            brep = ifc_file.create_entity("IfcFacetedBrep", Outer=face_set)

        shape_rep = ifc_file.create_entity("IfcShapeRepresentation",
            ContextOfItems=context,
            RepresentationIdentifier="Body",
            RepresentationType="Brep",
            Items=[brep]
        )

        add_global_style(ifc_file, brep)

        rep = ifc_file.create_entity("IfcProductDefinitionShape", Representations=[shape_rep])

        wall = ifc_file.create_entity("IfcBuildingElementProxy",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner_history,
            Name=f"Mur_{row['segment']}",
            Representation=rep,
            ObjectPlacement=site_placement
        )

        ifc_file.create_entity("IfcRelContainedInSpatialStructure",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner_history,
            RelatingStructure=storey,
            RelatedElements=[wall]
        )

    ifc_file.write(output_path)
    print(f"✅ Fichier IFC créé avec succès : {output_path}")
    return True


# def main_extrusion(file_ref):
#     altitude = get_building_storey_info(file_ref)
#     create_ifc_from_csv(
#         #csv_path="static/enveloppe_finale.csv",
#         csv_path="static/test.csv",
#         output_path="outputs/parcelle.ifc",
#         altitude = 43.7#altitude  # ou 0.0 selon ton besoin
#     )

# def ajouter_colonnes_csv(input_file="static/enveloppe_finale.csv", output_file="static/test.csv", type_val='VO', retrait_val=2):
#     # Lire le fichier CSV
#     df = pd.read_csv(input_file)
    
#     # Ajouter les colonnes avec les valeurs fixes
#     df['Type'] = type_val
#     df['Retrait'] = retrait_val
    
#     # Sauvegarder le nouveau fichier CSV
#     df.to_csv(output_file, index=False)
#     print(f"Fichier transformé sauvegardé sous : {output_file}")
def ajouter_colonnes_csv(analysis_id, type_val='VO', retrait_val=2):
    """
    Ajoute les colonnes 'Type' et 'Retrait' à la donnée JSON 'enveloppe_finale' 
    dans la table Analysis et met à jour la base de données.
    
    Args:
        analysis_id (int): ID de l'analyse concernée
        type_val (str): Valeur par défaut pour la colonne 'Type'
        retrait_val (float|int): Valeur par défaut pour 'Retrait'
    
    Returns:
        bool: True si succès, False sinon
    """
    analysis = Analysis.query.get(analysis_id)
    if not analysis or not analysis.enveloppe_finale:
        print(f"❌ Données introuvables pour l'analyse {analysis_id}")
        return False

    try:
        # Charger les données JSON dans un DataFrame
        df = pd.DataFrame(analysis.enveloppe_finale)

        # Ajouter les colonnes
        df['Type'] = type_val
        df['Retrait'] = retrait_val

        # Convertir à nouveau en JSON
        updated_json = df.to_dict(orient='records')
        # Sauvegarder dans la base
        analysis.enveloppe_finale = updated_json
        db.session.commit()

        print(f"✅ Colonnes ajoutées et enveloppe_finale mise à jour pour l'analyse {analysis_id}")
        return True

    except Exception as e:
        db.session.rollback()
        print(f"❌ Erreur lors de la mise à jour : {str(e)}")
        return False


import uuid
from datetime import datetime
import os

def generate_unique_filename(prefix: str = "parcelle", extension: str = "ifc") -> str:
    """Génère un nom de fichier unique avec timestamp et UUID.
    
    Format : {prefix}_{timestamp}_{uuid4_short}.{extension}
    
    Args:
        prefix: Préfixe descriptif (ex: "parcelle", "enveloppe")
        extension: Extension du fichier (sans le point)
        
    Returns:
        Nom de fichier unique (ex: "parcelle_20240514_3f2a1b.ifc")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]  # 6 premiers caractères du UUID
    return f"{prefix}_{timestamp}_{unique_id}.{extension}"

# def main_extrusion(analysis_id,file_ref, output_dir: str = "outputs"):
#     """Version améliorée avec gestion des fichiers sécurisée."""
#     # 1. Créer le dossier de sortie si inexistant
#     os.makedirs(output_dir, exist_ok=True)
#     print("DEBUT MAIN EXTRUSION")
    
#     # 2. Générer un nom de fichier unique
#     output_filename = generate_unique_filename(prefix="enveloppe", extension="ifc")
#     output_path = os.path.join(output_dir, output_filename)
    
#     # 3. Traitement principal
#     altitude = get_building_storey_info(file_ref)
#     ajouter_colonnes_csv(analysis_id)
#     create_ifc_from_csv(
#         #csv_path="static/test.csv",
#         analysis_id,
#         output_path=output_path,
#         altitude= 43.7  #altitude #si dynamique
#     )
    
#     return output_path  # Retourne le chemin complet pour référence



def main_extrusion(analysis_id,file_ref, output_dir: str = "static/outputs"):
    """Version améliorée avec gestion des fichiers sécurisée."""
    # 1. Créer le dossier de sortie si inexistant
    #os.makedirs(output_dir, exist_ok=True)
    print("DEBUT MAIN EXTRUSION")
    
    # 2. Générer un nom de fichier unique
    
    output_filename = generate_unique_filename(prefix="enveloppe", extension="ifc")
    analysis = Analysis.query.get(analysis_id)
    analysis_dir = "static/outputs/analyse_" + str(analysis.user_id) + "_"+ str(analysis_id) + "/"#os.path.join('static', 'analyses', str(analysis_id))
    os.makedirs(analysis_dir, exist_ok=True)
    chemin = analysis_dir + output_filename
    if not analysis.ifc_file:
        analysis.ifc_file = {}
    # S'assurer que c'est bien un dictionnaire, au cas où la DB contient une chaîne JSON par exemple
    if isinstance(analysis.ifc_file, str):
        import json
        analysis.ifc_file = json.loads(analysis.ifc_file)
    analysis.ifc_file['ifc'] = chemin
    db.session.commit()
    
    # 3. Traitement principal
    altitude = get_building_storey_info(file_ref)
    ajouter_colonnes_csv(analysis_id)
    create_ifc_from_csv(
        #csv_path="static/test.csv",
        analysis_id,
        output_path=chemin,
        altitude= 43.7  #altitude #si dynamique
    )
    
    return chemin  # Retourne le chemin complet pour référence