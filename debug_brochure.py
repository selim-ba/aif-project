import pickle
import sys
from pathlib import Path

# Chemin vers votre fichier brochure
path = Path("models/part4_rag_brochure.pkl")

if not path.exists():
    print(f"❌ ERREUR : Le fichier {path} n'existe pas !")
    sys.exit(1)

print(f"📂 Lecture du fichier : {path}")

try:
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    print(f"✅ Fichier chargé avec succès.")
    print(f"📊 Nombre total de films en mémoire : {len(data)}")
    
    if len(data) == 0:
        print("⚠️ ALERTE : La brochure est VIDE !")
        sys.exit(1)

    # On prend le tout premier film pour examiner ses "organes"
    first_key = list(data.keys())[0]
    first_movie = data[first_key]
    
    print("\n" + "="*40)
    print("🔎 AUTOPSIE DU PREMIER FILM")
    print("="*40)
    print(f"🔑 Clé (ID utilisé par le système) : {first_key}")
    print(f"Type de la clé : {type(first_key)}")
    print("-" * 20)
    print("CONTENU EXACT DES CHAMPS :")
    
    for key, value in first_movie.items():
        # On affiche le nom exact de la colonne (key) et un aperçu du contenu
        preview = str(value)[:80] + "..." if isinstance(value, str) else str(value)
        print(f"👉 NOM COLONNE: '{key}' \t CONTENU: {preview}")

    print("="*40)
    print("Copiez-collez ce résultat dans la conversation !")

except Exception as e:
    print(f"❌ Erreur lors de la lecture : {e}")