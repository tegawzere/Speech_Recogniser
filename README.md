# Simple Speech Recognizer

A MATLAB-based speech recognizer using Hidden Markov Models (HMMs) trained on a small vocabulary. This project includes feature extraction, HMM training, and evaluation with a confusion matrix. The implementation supports a vocabulary of eleven words and includes example training and testing data.

## Project Overview

This project uses a simple HMM-based speech recognizer model, implemented in MATLAB, with a focus on feature extraction using Mel-Frequency Cepstral Coefficients (MFCCs), training with the Baum-Welch algorithm, and sequence prediction using the Viterbi algorithm. 

### Features
- **Vocabulary**: Recognizes a set of eleven words.
- **Feature Extraction**: Uses 13-dimensional MFCCs.
- **Training**: Baum-Welch re-estimation on HMMs.
- **Evaluation**: Viterbi decoding, accuracy measurement, and confusion matrix visualization.

## File Structure

- **data/**: Contains example training, testing, and evaluation datasets.
- **src/**: MATLAB scripts for feature extraction, HMM training, and evaluation.
- **results/**: Stores generated results, including confusion matrix images and accuracy metrics.
- **docs/**: Summary report and additional documentation.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/username/speech-recognizer.git
    ```
2. Open MATLAB and navigate to the `src` folder.

## Usage

### 1. Feature Extraction

Run the `feature_extraction.m` script to extract MFCCs from the audio data:
```matlab
feature_extraction('data/training_data');
