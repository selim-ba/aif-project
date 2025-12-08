import gradio as gr
import requests
import json
import os

# --- Fonction de prédiction qui appelle votre API Flask ---
def predict_poster_genre_gradio(image_file):
    """
    Fonction appelée par Gradio pour envoyer l'image à l'API Flask
    et récupérer les prédictions.
    """
    if image_file is None:
        return "Veuillez télécharger une image.", []

    try:
        # Ouvrir le fichier image en mode binaire
        # 'rb' = read binary
        with open(image_file, 'rb') as f:
            # Préparer le dictionnaire 'files' pour la requête requests
            # La clé 'file' doit correspondre à ce que votre API Flask attend (request.files['file'])
            files = {'file': (os.path.basename(image_file), f, 'image/jpeg')}
            
            # Envoyer la requête POST à votre API Flask
            response = requests.post("http://127.0.0.1:8000/api/predict_poster_genre", files=files)
            
            # Vérifier si la requête a réussi (statut 200)
            if response.status_code == 200:
                result = response.json()
                
                # Formatter les prédictions pour l'affichage Gradio
                if "predictions" in result and isinstance(result["predictions"], list):
                    formatted_predictions = []
                    for pred in result["predictions"]:
                        genre = pred.get("genre", "N/A")
                        score = pred.get("score", 0.0)
                        formatted_predictions.append(f"{genre}: {score:.3f}")
                    return "Prédictions réussies:", formatted_predictions
                else:
                    return f"Réponse inattendue de l'API: {result}", []
            else:
                # Gérer les erreurs de l'API Flask (ex: 400 Bad Request, 500 Internal Server Error)
                error_message = response.json().get("error", f"Erreur inconnue ({response.status_code})")
                return f"Erreur de l'API Flask: {error_message}", []

    except requests.exceptions.ConnectionError:
        return "Erreur de connexion: Assurez-vous que votre API Flask est en cours d'exécution à l'adresse http://127.0.0.1:8000/api/predict_poster_genre", []
    except Exception as e:
        return f"Une erreur inattendue est survenue: {e}", []

# --- Définition de l'interface Gradio ---
# Input: un composant gr.Image pour le téléchargement de fichiers
# Output: un composant gr.Label pour le texte des prédictions
#         un composant gr.JSON pour la structure complète (utile pour le debug)
interface = gr.Interface(
    fn=predict_poster_genre_gradio,
    inputs=gr.Image(type="filepath", label="Téléchargez l'affiche de film"),
    outputs=[
        gr.Markdown(label="Statut"),
        gr.JSON(label="Prédictions Détaillées")
    ],
    title="🎬 Prédicteur de Genre de Film par Affiche",
    description="Téléchargez une affiche de film, et l'API prédira les genres associés."
)

# --- Lancer l'interface Gradio ---
if __name__ == "__main__":
    # Pour exécuter, utilisez: python gradio-app.py
    # Laisser share=True pour obtenir un lien public temporaire (utile pour partager)
    # Laissez share=False pour une utilisation locale uniquement.
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)