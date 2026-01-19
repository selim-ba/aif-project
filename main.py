from fastapi import FastAPI
from app.plot.schemas import PlotRequest, GenreResponse
from app.plot.model import predict_genre_logic

app = FastAPI(title="Movie Genre & Recommendation API")

@app.post("/predict_genre", response_model=GenreResponse)
async def predict_genre(request: PlotRequest):
    """
    Predicts the genre of a movie based on its plot description.
    """
    result = predict_genre_logic(request.plot)
    return result

# Pour lancer : uvicorn main:app --reload