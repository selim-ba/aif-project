# gradio-app.py

import gradio as gr
import requests
import os

# API base URL - uncomment the appropriate one
API_URL = "http://127.0.0.1:8000"  # local testing
# API_URL = "http://host.docker.internal:8000"  # docker on Mac/Windows
# API_URL = "http://flask-api:8000"  # docker-compose / cloud


# Part 1&2 : Poster Genre Prediction
def predict_poster_genre_gradio(image_file):
    """Send image to Flask API for genre prediction."""
    if image_file is None:
        return "Please upload an image.", []

    try:
        with open(image_file, 'rb') as f:
            files = {'file': (os.path.basename(image_file), f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/api/predict_poster_genre", files=files)

            if response.status_code == 200:
                result = response.json()
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
                error_message = response.json().get("error", f"Unknown error ({response.status_code})")
                return f"Flask API error: {error_message}", []
    except Exception as e:
        return f"An error occurred: {e}", []


def check_poster(image):
    """Check if the uploaded image is a valid movie poster."""
    if image is None:
        return "Please upload an image."

    try:
        with open(image, 'rb') as f:
            files = {'file': (os.path.basename(image), f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/api/check_is_poster", files=files)

            if response.status_code == 200:
                data = response.json()
                status = "It's a poster!" if data['is_poster'] else "ALERT: This is not a poster."
                return f"{status}\nAnomaly score: {data['anomaly_score']:.4f}"
            else:
                return f"API Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"
    
def predict_genre_from_plot(plot_text):
    if not plot_text:
        return "Please enter a movie plot."
    
    try:
        response = requests.post(
            f"{API_URL}/api/predict_plot_genre", 
            json={"plot": plot_text}
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            return {"Info": "NLP model not loaded. Please try again later."}
        else:
            return {"Error": f"API {response.status_code}"}
    except Exception as e:
        return {"Error": str(e)}


# Par 4: NLP/RAG Movie Chat
def chat_with_movies(message, conversation_history):
    """Send a message to the RAG chat API and get a response."""
    if not message.strip():
        return conversation_history, "", "Please enter a message."

    try:
        response = requests.post(
            f"{API_URL}/api/chat",
            json={"query": message},
            headers={"Content-Type": "application/json"},
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                bot_response = data.get("response", "No response received.")
            else:
                bot_response = f"Error: {data.get('error', 'Unknown error')}"
        elif response.status_code == 503:
            bot_response = "RAG service not available. Please check if the model is loaded."
        else:
            bot_response = f"API Error ({response.status_code}): {response.text}"

    except requests.exceptions.Timeout:
        bot_response = "Request timed out. The model might be loading."
    except requests.exceptions.ConnectionError:
        bot_response = "Cannot connect to the API. Is the server running?"
    except Exception as e:
        bot_response = f"Error: {str(e)}"

    # Update conversation history as plain text
    new_history = conversation_history + f"\n\nYOU: {message}\n\nASSISTANT: {bot_response}"
    return new_history.strip(), "", "Response received!"


def reset_conversation():
    """Reset the RAG conversation history."""
    try:
        requests.post(f"{API_URL}/api/reset_chat", timeout=10)
        return "", "Conversation reset! Start a new chat."
    except Exception as e:
        return "", f"Error: {str(e)}"


# Gradio Interface
with gr.Blocks(title="INSA Toulouse Project : AI Tools for a Movie Streaming Platform") as demo:
    gr.Markdown("""
    # AI Tools for a Movie Streaming Platform
    ### Group Members : Selim Ben Abdallah, Arman Hosseini, Paul Slisse, Guillaume Staub
    
    Choose a feature below:
    - **Poster Analysis**: Check if an image is a valid poster and predict its genre.
    - **Plot-based genre prediction**: Predict the genre of a movie based on its plot.
    - **Natural Language Movie Discovery**: Chat with AI to find movies you'll might like.
    """)

    with gr.Tabs():
        # Poster Analysis (Parts 1-2)
        with gr.TabItem("Poster Analysis"):
            gr.Markdown("### Upload a movie poster to analyze")

            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="filepath", label="Upload Movie Poster")

                    with gr.Row():
                        btn_check = gr.Button("1. Check if it's a poster", variant="secondary")
                        btn_predict = gr.Button("2. Predict genre", variant="primary")

                with gr.Column():
                    output_check = gr.Textbox(label="Validation Result", lines=3)
                    output_predict = gr.JSON(label="Genre Predictions")

            btn_check.click(fn=check_poster, inputs=input_image, outputs=output_check)
            btn_predict.click(fn=predict_poster_genre_gradio, inputs=input_image, outputs=output_predict)
        
        # Plot-based genre prediction Genre Prediction (Part 3)
        with gr.TabItem("Plot-based genre prediction"):
            gr.Markdown("### Predict the genre of a movie via its summary")
            with gr.Row():
                with gr.Column():
                    plot_input = gr.Textbox(lines=5, label="Movie plot", placeholder="A group of heroes saves the world...")
                    nlp_btn = gr.Button("Predict the genre", variant="primary")
                with gr.Column():
                    nlp_output = gr.JSON(label="Genre predictions")
            
            nlp_btn.click(predict_genre_from_plot, inputs=plot_input, outputs=nlp_output)

        # Natural Language Movie Discovery Chat (Part 4)
        with gr.TabItem("Natural Language Movie Discovery"):
            gr.Markdown("""
            ### Chat with AI to discover movies you might like !
            """)

            conversation_box = gr.Textbox(
                label="Conversation",
                lines=15,
                max_lines=20,
                interactive=False,
                placeholder="Your conversation will appear here..."
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Describe what kind of movie you want to watch...",
                    lines=2,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("Reset Conversation", variant="secondary")
                status_text = gr.Textbox(label="Status", interactive=False, scale=2)

            # gr.Markdown("""
            # **Try these examples:**
            # - I want a sci-fi movie with a cyberpunk aesthetic
            # - Something funny for a family movie night
            # - A thriller with a strong female lead
            # - Movies about time travel with plot twists
            # """)

            msg_input.submit(
                chat_with_movies,
                [msg_input, conversation_box],
                [conversation_box, msg_input, status_text]
            )
            send_btn.click(
                chat_with_movies,
                [msg_input, conversation_box],
                [conversation_box, msg_input, status_text]
            )
            clear_btn.click(reset_conversation, outputs=[conversation_box, status_text])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)