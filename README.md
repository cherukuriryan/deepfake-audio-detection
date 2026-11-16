# Deepfake Audio Detection

## Synthetic Deception at Scale

A senior capstone project focused on detecting AI-generated and manipulated speech using machine learning and audio feature analysis.

### Team Members

- **Ryan Cherukuri**
- **Pedro Magalhaes Brito Scardazzi**
- **Isaac Euceda-Colindres**

---

## Project Overview

This project investigates the challenge of detecting synthetic and deepfake audio. Our goal was to build a machine learning system that can analyze an uploaded audio recording and classify it as either genuine human speech or synthetic/spoofed speech.

During development, we found that high benchmark accuracy did not always translate to strong real-world performance. Because of this, the project expanded beyond simply improving accuracy and focused on:

- Cross-dataset generalization
- Domain shift
- Feature representation
- Model confidence
- Real-world microphone recordings
- Multiple machine learning approaches

The final system includes:

- Audio preprocessing with Librosa
- 130 acoustic features per recording
- A trained Multi-Layer Perceptron classifier
- Cross-dataset evaluation
- A Flask backend
- A browser-based audio upload interface

---

## Detection Pipeline

**Audio Upload → Preprocessing → Feature Extraction → MLP Model → Prediction + Confidence Score**

Audio is loaded as:

- Mono
- 16 kHz sample rate

The final model extracts **130 acoustic features** from each recording.

These include:

- 20 MFCC means
- 20 MFCC standard deviations
- 20 delta-MFCC means
- 20 delta-MFCC standard deviations
- 20 delta-delta MFCC means
- 20 delta-delta MFCC standard deviations
- Spectral centroid mean and standard deviation
- Spectral bandwidth mean and standard deviation
- Spectral rolloff mean and standard deviation
- Zero-crossing rate mean and standard deviation
- RMS energy mean and standard deviation

---

## Datasets

### DFADD

Our primary dataset was the Deepfake Audio Detection Dataset.

- Hugging Face: `isjwdu/DFADD`
- Approximately 207,955 samples
- Real and spoofed speech
- Primarily 16 kHz FLAC audio

### ASVspoof 2019 LA

ASVspoof was added to increase training diversity and test whether the detector could learn from more than one audio domain.

### In-the-Wild

The In-the-Wild dataset was used as our strongest independent cross-dataset evaluation set.

This helped us measure whether the model could generalize beyond the datasets used during training.

---

## Models Tested

Throughout the project, we tested several machine learning approaches:

- Random Forest
- Support Vector Machine
- Gradient Boosting
- Logistic Regression
- Multi-Layer Perceptron
- Extra Trees
- K-Nearest Neighbors
- Multi-dataset MLP training
- Audio augmentation
- WavLM embeddings with Logistic Regression

One of our main findings was that simply changing classifiers did not solve the cross-dataset generalization problem.

The largest improvement came from creating a richer acoustic feature representation.

---

## Final Model

Our final deployment model is a:

**Richer Feature Multi-Layer Perceptron**

Training data:

- 4,000 real DFADD samples
- 4,000 spoofed DFADD samples
- 4,000 real ASVspoof samples
- 4,000 spoofed ASVspoof samples

Total:

**16,000 balanced samples**

Feature size:

**130 features per sample**

---

## Key Results

| Experiment | Accuracy |
|---|---:|
| Balanced DFADD MLP with 20 MFCC features | 95.65% |
| Original MLP on In-the-Wild | 53.00% |
| Original MLP on ASVspoof | 50.00% |
| Equal-source multi-dataset MLP | 96.06% |
| Equal-source MLP on In-the-Wild | 55.00% |
| **Richer 130-feature MLP** | **99.75%** |
| **Richer MLP on In-the-Wild** | **69.50%** |
| **Richer MLP on ASVspoof evaluation** | **81.00%** |
| WavLM + Logistic Regression pilot | 99.50% held-out |

The richer MLP produced the strongest overall combination of internal performance and cross-dataset generalization.

---

## Main Research Finding

Our most important finding was:

> **High benchmark accuracy does not guarantee strong real-world deepfake detection performance.**

Several models achieved approximately 95–99% accuracy on held-out samples from familiar datasets but performed much worse on unfamiliar recordings.

The biggest challenge was **domain shift**.

Differences in:

- Microphones
- Recording environments
- Audio codecs
- Signal processing
- Dataset sources
- Synthetic speech generators

can significantly affect model predictions.

Fresh genuine microphone recordings were sometimes incorrectly classified as synthetic with extremely high confidence.

Because of this, the model confidence score should not be interpreted as absolute certainty.

---

## Web Application

The repository includes a Flask web application that allows users to upload an audio file and receive a prediction.

The interface displays:

- Likely Real or Likely Synthetic
- Model confidence score
- Model classification
- Research prototype disclaimer

The website uses the same trained richer MLP and feature extraction pipeline as the command-line predictor.

---

## Running the Project

### Clone the repository

```bash
git clone https://github.com/cherukuriryan/deepfake-audio-detection.git
cd deepfake-audio-detection
```

### Create a virtual environment

```bash
python3 -m venv .venv
```

### Activate the virtual environment

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Start the Flask application

```bash
python app.py
```

You should see:

```text
Running on http://127.0.0.1:5000
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

Keep the terminal running while using the website.

---

## Running the Model Directly

The detector can also be used from the terminal without the website.

```bash
python src/predict_audio.py path/to/audio_file.wav
```

Example output:

```text
Prediction Results
--------------------
Prediction: spoofed
Confidence: 92.75%
```

---

## Project Structure

```text
deepfake-audio-detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── deepfake_audio_richer_multidataset_mlp.pkl
│
├── src/
│   ├── predict_audio.py
│   ├── train_richer_multidataset_mlp.py
│   ├── build_richer_multidataset_features.py
│   ├── train_richer_extra_trees.py
│   ├── train_wavlm_logistic.py
│   ├── build_wavlm_features.py
│   └── additional experiment scripts
│
└── Deepfake Detection Site Prototype/
    ├── home.html
    ├── results.html
    ├── loading.html
    └── custom.css
```

---

## Limitations

This detector is an experimental research prototype.

Known limitations include:

- Reduced performance on unfamiliar recording devices
- Reduced cross-dataset performance
- False positives on genuine microphone recordings
- Confidence scores are not calibrated probabilities
- Limited real-world microphone testing
- Limited computing resources
- Training data cannot represent every possible synthetic speech generator
- The model should not be used as forensic proof

Results should always be interpreted carefully.

---

## Future Work

Future improvements could include:

- More real microphone recordings
- Additional recording devices and environments
- More synthetic speech generators
- Additional speakers and languages
- Domain adaptation
- Confidence calibration
- Out-of-distribution detection
- Fine-tuning pretrained speech models such as WavLM
- wav2vec 2.0
- HuBERT
- Spectrogram-based neural networks
- Transformer-based temporal models
- Ensemble methods

---

## Technologies Used

- Python
- Librosa
- NumPy
- Scikit-learn
- Joblib
- Hugging Face Datasets
- Flask
- HTML
- CSS
- JavaScript
- Git
- GitHub

---

## Disclaimer

This project was developed for academic and research purposes.

The detector is not intended to provide definitive proof that an audio recording is real or synthetic. Predictions may be incorrect, especially when audio comes from recording devices, environments, codecs, or datasets that differ from the training data.

## Disclaimer

This system is being developed as an academic prototype. A classification produced by the system should not be treated as definitive proof that an audio recording is authentic or AI-generated.
