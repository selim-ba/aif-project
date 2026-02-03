import pandas as pd
import pickle
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"Lecture de {args.csv}...")
    df = pd.read_csv(args.csv)
    
    movies_db = {}
    
    # On parcourt le CSV pour créer le dictionnaire
    for idx, row in df.iterrows():
        # Extraction basique du titre
        plot = str(row.get("movie_plot", ""))
        title = f"Movie {idx}" # Vous pouvez améliorer l'extraction de titre ici si besoin
        
        movies_db[idx] = {
            "title": title,
            "plot": plot,
            "poster_path": str(row.get("movie_poster_path", "")),
            "category": str(row.get("movie_category", ""))
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(movies_db, f)
        
    print(f"✅ Fichier '{args.out}' généré ({len(movies_db)} films).")

if __name__ == "__main__":
    main()