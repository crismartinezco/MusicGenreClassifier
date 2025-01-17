# :musical_note: Music Genre Classifier
Implementation of two different ML models; a multilayer perceptron and a convolutional neural network, for music genre classification using the GTZAN Dataset (https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).  A performance comparison is performed, which is thoroughly explaned in https://www.kaggle.com/code/crismartinezco/different-models-for-music-genre-classification/edit.

## Table of Contents
- [Installation](#installation)

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo.git](https://github.com/crismartinezco/MusicGenreClassifier)

2. Navigate to the project directory:

cd your-repo

3. Install dependencies:

pip install -r requirements.txt

- [Usage](#usage)

The models are deployed with uvicorn, although any service can work by exporting the models. To deploy the code straight after installation run the following code:

```bash
uvicorn MGCDeployment:app --host 0.0.0.0 --port 8000

```bash
jupyter lab

Run the MCG GUI notebook

- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

