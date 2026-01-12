# Model Weights

Model weights are **NOT** stored in this repository.

They are hosted on Google Drive.

## Part 1 – Poster genre model

- File: `movie_genre_cpu.pt`
- Location: https://drive.google.com/drive/folders/16oitro8r5frXZE3N1jsFDzxZXDaghliw?usp=sharing

For local development:

1. Download `movie_genre_cpu.pt` from the link above.
2. Place it in this `models/` directory so the full path is:

   `models/movie_genre_cpu.pt`

## Part 2 – Detection of false poster model

- File: `ood_detector.joblib`
- Location: https://drive.google.com/drive/folders/16oitro8r5frXZE3N1jsFDzxZXDaghliw?usp=sharing

For local development:

1. Download `ood_detector.joblib` from the link above.
2. Place it in this `models/` directory so the full path is:

   `models/ood_detector.joblib`


## Part 3 – Predicting genre from plots

- Files: `part3_movie_weights.pth`, `part3_movie_brochure.pkl`, `part3_movie_index.ann`
- Location: https://drive.google.com/drive/folders/16oitro8r5frXZE3N1jsFDzxZXDaghliw?usp=sharing

For local development:

1. Download `part3_movie_weights.pth`, `part3_movie_brochure.pkl`, `part3_movie_index.ann` from the link above.
2. Place it in this `models/` directory so the full path is:

   `models/part3_movie_weights.pth`
   `models/part3_movie_brochure.pkl`
   `models/part3_movie_index.ann`

Description:

 - `part3_movie_weights.pth` is the file containing the weights of the model (NLP model).
 - `part3_movie_brochure.pkl` is a Pickle file, the dataset in a binary form to be processed faster by Python.
 - `part3_movie_index.ann` is the Annoy index, used to do the link between the query and the dataset.
