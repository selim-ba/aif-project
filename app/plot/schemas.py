#schemas.py: to hold the classes necessitated in the API.

from pydantic import BaseModel

class PlotRequest(BaseModel):
    plot: str

class GenreResponse(BaseModel):
    predicted_genre: str
    label_id: int