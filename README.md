# Synthetic Deception at Scale

## Audio Deepfake Detection System

### Overview

Generative AI has made it increasingly easy to create realistic synthetic media, including images, videos, text, and audio. While these technologies have many useful applications, they can also be used to create deceptive or non-consensual content.

This senior capstone project examines the ethical, political, and interpersonal harms associated with generative AI while developing an audio deepfake detection system. The software component of the project is designed to analyze speech recordings and classify them as either authentic or AI-generated.

## Project Goals

The goals of this project are to:

- Research the ethical and privacy concerns surrounding generative AI.
- Examine the use of synthetic media in misinformation and political content.
- Explore how AI-generated media can affect interpersonal trust.
- Develop a system capable of distinguishing AI-generated speech from authentic speech.
- Provide users with a classification result and confidence score.
- Evaluate the accuracy and limitations of audio deepfake detection.

## How the System Works

The proposed detection system follows this general process:

1. The user uploads an audio file.
2. The audio is preprocessed into a consistent format.
3. Relevant audio features are extracted.
4. A trained classification model analyzes the features.
5. The system predicts whether the audio is **Real** or **AI-Generated**.
6. The classification and confidence score are displayed to the user.

## Technologies

The project is being developed using:

- **Python** — Audio processing and machine learning
- **Librosa** — Audio analysis and feature extraction
- **Scikit-learn** — Model training and classification
- **Flask** — Backend and web application
- **HTML/CSS/JavaScript** — User interface
- **Git/GitHub** — Version control and team collaboration

## Dataset

The detection model will be trained using datasets containing both authentic and AI-generated speech samples.

Audio files will be preprocessed and divided into training and testing data so that the performance of the classification model can be evaluated on audio that was not used during training.

Dataset information and citations will be added as datasets are finalized.

## Model Development

The initial model will use audio features extracted with Librosa to train a machine-learning classifier using Scikit-learn.

The model will be evaluated using measures such as:

- Classification accuracy
- Precision
- Recall
- F1-score
- False positives
- False negatives

Additional models and audio features may be evaluated as development continues.

## Project Structure

The planned project structure is:

    deepfake-audio-detection/
    │
    ├── data/
    ├── models/
    ├── src/
    ├── web/
    ├── tests/
    ├── docs/
    ├── requirements.txt
    └── README.md



All team members will contribute to system integration, testing, research, documentation, and the final presentation.

## Project Timeline

- **September 9:** 30% coding complete
- **September 30:** 60% coding complete
- **October 21:** 90% coding complete
- **November 13:** Coding complete and library poster
- **December 2:** Final presentation and report

## Current Status

This project is currently under development. The team is working on dataset selection, audio preprocessing, feature extraction, model development, and the initial web application.

## Disclaimer

This system is being developed as an academic prototype. A classification produced by the system should not be treated as definitive proof that an audio recording is authentic or AI-generated.
