from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import librosa
import numpy as np
import io

app = FastAPI()

# Load the trained Keras models for each type
overf_mp = load_model("path\\to\\overfittedmp")
mp = load_model("path\\to\\mp")
cnn = load_model("path\\to\\cnn")

def preprocess_wav(file):
    # Load the audio file
    signal, sr = librosa.load(io.BytesIO(file), sr=22050, duration=30)  # Ensure consistent sampling rate

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T  # Transpose to get the correct input shape

    # Adjust size if necessary (pad or truncate)
    target_length = 130  # This should match the length your model expects
    if mfcc.shape[0] < target_length:
        # Pad with zeros if mfcc is shorter
        mfcc = np.pad(mfcc, ((0, target_length - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        # Truncate if mfcc is longer
        mfcc = mfcc[:target_length, :]

    # Ensure the final shape is correct (batch size, time steps, features)
    return np.expand_dims(mfcc, axis=0)

# List of genres for reference
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

@app.post("/predict_overfitted_mlp")
async def predict_overfitted_mlp(file: UploadFile = File(...)):
    # Read and preprocess the uploaded file
    file_bytes = await file.read()
    input_data = preprocess_wav(file_bytes)

    # Make prediction using the overfitted MLP model
    prediction = overf_mp.predict(input_data)

    # Get the predicted genre index
    genre_index = np.argmax(prediction, axis=1)[0]  # Get the index of the genre with the highest probability

    # Get the confidence (accuracy) as the probability associated with the predicted genre
    predicted_confidence = float(np.max(prediction))  # This is the maximum probability of the predicted genre

    predicted_genre = genres[genre_index]

    return {"predicted_genre": predicted_genre, "confidence": predicted_confidence}


@app.post("/predict_mlp_no_overfit")
async def predict_mlp_no_overfit(file: UploadFile = File(...)):
    # Read and preprocess the uploaded file
    file_bytes = await file.read()
    input_data = preprocess_wav(file_bytes)

    # Make prediction using the MLP model without overfitting
    prediction = mp.predict(input_data)

    # Get the predicted genre index
    genre_index = np.argmax(prediction, axis=1)[0]

    # Get the confidence
    predicted_confidence = float(np.max(prediction))

    predicted_genre = genres[genre_index]

    return {"predicted_genre": predicted_genre, "confidence": predicted_confidence}


@app.post("/predict_cnn")
async def predict_cnn(file: UploadFile = File(...)):
    # Read and preprocess the uploaded file
    file_bytes = await file.read()
    input_data = preprocess_wav(file_bytes)

    # Make prediction using the CNN model
    prediction = cnn.predict(input_data)

    # Get the predicted genre index
    genre_index = np.argmax(prediction, axis=1)[0]

    # Get the confidence
    predicted_confidence = float(np.max(prediction))

    predicted_genre = genres[genre_index]

    return {"predicted_genre": predicted_genre, "confidence": predicted_confidence}

