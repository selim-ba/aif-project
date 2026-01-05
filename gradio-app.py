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
            #response = requests.post("http://127.0.0.1:8000/api/predict_poster_genre", files=files) #local
            #response = requests.post("http://host.docker.internal:8000/api/predict_poster_genre", files=files) #via docker
            response = requests.post("http://flask-api:8000/api/predict_poster_genre", files=files) #via cloud
            
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
                    return "Predictions successful:", formatted_predictions
                else:
                    return f"Unexpected API response: {result}", []
            else:
                # Gérer les erreurs de l'API Flask (ex: 400 Bad Request, 500 Internal Server Error)
                error_message = response.json().get("error", f"Unknown error ({response.status_code})")
                return f"Flask API error: {error_message}", []

    except Exception as e:
        return f"An error occurred: {e}", []
    
def check_poster(image):
    if image is None:
        return "Please upload an image."
    
    try:
        with open(image, 'rb') as f:
            files = {'file': (os.path.basename(image), f, 'image/jpeg')}
            #response = requests.post("http://127.0.0.1:8000/api/check_is_poster", files=files) #local
            #response = requests.post("http://host.docker.internal:8000/api/check_is_poster", files=files) #via docker
            response = requests.post("http://flask-api:8000/api/check_is_poster", files=files) #via cloud

            if response.status_code == 200:
                data = response.json()
                status = "✅ It's a poster !" if data['is_poster'] else "🚨 ALERT : This is not a poster."
                return f"{status}\nAnomaly score : {data['anomaly_score']:.4f}"
            else:
                return f"API Error : {response.text}"
    except Exception as e:
        return f"Error : {str(e)}"
    
# --- Construction de l'interface avec Blocks ---
with gr.Blocks(title="Movie Poster AI") as demo:
    gr.Markdown("# 🎬 Movie Poster AI")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Your Movie Poster Image")
            
            # --- Les deux boutons ---
            with gr.Row():
                btn_check = gr.Button("1. Check if it's a poster", variant="secondary")
                btn_predict = gr.Button("2. Predict genre", variant="primary")
        
        with gr.Column():
            # Sorties différentes pour chaque action
            output_check = gr.Textbox(label="Validation Result")
            output_predict = gr.JSON(label="Predictions Result")

    # --- Connexion des événements ---
    # Quand on clique sur "Vérifier", on appelle check_poster et on affiche dans output_check
    btn_check.click(fn=check_poster, inputs=input_image, outputs=output_check)
    
    # Quand on clique sur "Prédire", on appelle predict_genre et on affiche dans output_predict
    btn_predict.click(fn=predict_poster_genre_gradio, inputs=input_image, outputs=output_predict)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)