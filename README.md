# :musical_note: Music Genre Classifier
Implementation of two different end-to-end ML models; a multilayer perceptron (including an approach to solving overfitting) and a convolutional neural network, for music genre classification using the GTZAN Dataset (https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/crismartinezco/MusicGenreClassifier

2. Navigate to the project directory:
   ```bash
   cd your-repo

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

The models are deployed with uvicorn, although any service can work by exporting the models. 

1. To deploy the code straight after installation run the following code:
   ```bash
   uvicorn MGCDeployment:app --host 0.0.0.0 --port 8000

2. Then run jupyter lab:
   ```bash
   jupyter lab

3. Run the MCG GUI notebook.

## Features

- The models are trained based on the MFCC extracted using librosa. A performance comparison of the models is presented in https://www.kaggle.com/code/crismartinezco/different-models-for-music-genre-classification/edit.
- As an end-to-end model, necessary functions for running predictions upon new data are provided.

## Known Issues

Here are some known bugs and limitations in this project:

1. **Format and length of the user provided audio data**  
   - **Description**: Only .wav with a max. length of 30secs. are analyzed. If longer files are provided, no prediction is performed.
   - **Workaround**: Load 30s. long .wav files. To cut the files use any online daw like https://www.soundtrap.com/de/musicmakers.

2. **Error 404 on GUI**  
   - **Description**: Error 404 when trying to run different predictions with the same uploaded file.
   - **Workaround**: For running the predictions from each model on the GUI, the file has to be uploaded each time.

## License

