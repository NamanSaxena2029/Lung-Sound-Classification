# Respiratory Disease Prediction System

## 🚀 Live Demo

🔗 **Try the application here:**
https://lungdiseaseclassification.streamlit.app/

An end-to-end AI project for lung disease prediction using respiratory sound recordings and a hybrid CNN+LSTM deep learning model. This repository covers everything from environment setup, data preparation, model training, to deploying a modern Streamlit web app for real-time predictions.

---

## Project Setup

### 1. Clone the repository

```bash
git clone https://github.com/NamanSaxena2029/Lung-Sound-Classification.git
cd Lung-Sound-Classification
```

### 2. Install dependencies

```bash
pip install streamlit librosa tensorflow matplotlib numpy pandas scikit-learn seaborn gdown
```

### 3. Dataset

* [Download the ICBHI 2017 Respiratory Sound Dataset from Kaggle](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)
* Place the extracted respiratory sound dataset in the `extracted_dataset/` folder.
* Ensure `patient_diagnosis.csv` and audio files are present.

### 4. Model File

* After training, save your model as `final_model.h5` in the project root directory.
* The app will **auto-download** the model from Google Drive if `final_model.h5` is not found locally.

---

## Environment & Imports

All code is written in Python. Key libraries used:

* `numpy`, `pandas` for data handling
* `librosa` for audio feature extraction and visualization
* `matplotlib` for waveform, spectrogram, and feature plots
* `tensorflow.keras` for deep learning (CNN, LSTM)
* `scikit-learn` for preprocessing, metrics, and cross-validation
* `streamlit` for the web interface
* `gdown` for auto-downloading the trained model

---

## Data Preparation & Feature Extraction

1. **Read and organize patient diagnosis and audio files**

2. **Extract features** from each audio file:

   * MFCC coefficients (40)
   * Chroma features
   * Mel spectrogram
   * Spectral contrast
   * Tonnetz features

3. **Data augmentation** for minority classes to balance the dataset

4. **Label encoding** for disease classes

---

## Model Architecture & Training

### 1. CNN + LSTM Hybrid Model

* 1D CNN layers for feature extraction
* LSTM layer for temporal pattern learning
* Dense layers for classification

### 2. Training

* K-Fold cross-validation for robust evaluation
* Early stopping to prevent overfitting
* Learning rate reduction during plateau
* Model checkpointing to save the best model
* Class weights to handle dataset imbalance

### 3. Final Model

* Train on the entire dataset
* Save the trained model as `final_model.h5`

---

## Results

| Metric    | Score    |
| --------- | -------- |
| Accuracy  | **91%**  |
| Precision | **0.92** |
| Recall    | **0.91** |
| F1 Score  | **0.92** |

### Training Performance

![Accuracy vs Loss graph](Images/Accuracy%20vs%20Loss%20graph.png)

### Confusion Matrix

![Confusion Matrix](Images/Confusion%20Matrix.png)

### ROC Curve

![ROC Curve](Images/ROC%20Curve.png)

---

## Streamlit Web App

### 1. Modern UI with sidebar navigation

The sidebar includes:
* Model information (CNN + LSTM architecture details)
* Supported audio formats (WAV, MP3, M4A)
* Step-by-step usage instructions
* Medical disclaimer notice

![Streamlit Sidebar](Images/Sidebar%20Section.png)

### 2. Upload audio (WAV, MP3, M4A)

* Upload any respiratory sound recording
* File name and size are displayed after upload
* In-browser audio playback available

![Upload Section](Images/Audio%20Upload%20and%20Playback.png)

### 3. Tabbed Results Interface

After uploading, results are shown in **two tabs**:

---

#### 🩺 Tab 1 — Prediction Results

* **Primary disease prediction** with confidence score
* **Disease description** for the predicted condition
* **Confidence distribution** bar chart across all 6 classes
* **Detailed probability table** with High/Low confidence labels
* **Recommendations** with health tips for the predicted disease

![Prediction Output](Images/Prediction%20Output.png)

![Prediction Output](Images/Prediction%20Output%202.png)

* **Medical disclaimer and educational purpose note**

![Disclaimer](Images/Disclaimer.png)
---

#### 📊 Tab 2 — Audio Analysis

All visualizations are displayed in a **2-column grid layout**:

| Visualization | Description |
|---|---|
| 🎵 Audio Waveform | Amplitude over time |
| 🔊 Mel Spectrogram | Frequency-time energy map (magma colormap) |
| 🎼 MFCC Features | 40 MFCC coefficients heatmap (coolwarm) |
| 🎹 Chroma Features | Pitch class energy over time (viridis) |
| 📉 Spectral Contrast | Frequency band contrast (plasma) |
| 🎶 Tonnetz Features | Tonal centroid features (RdYlGn) |
| 📈 Prediction Confidence | Progress bar showing confidence % |
| ⏱ Processing Time | Time taken + sample rate + audio duration |

---

## How to Run

1. Start the application:

```bash
streamlit run lung_disease_predictor.py
```

2. Open the provided **local URL** in your browser
3. Upload a lung sound file (WAV / MP3 / M4A)
4. View prediction results in **Tab 1** and audio feature analysis in **Tab 2**

---

## Supported Disease Classes

| Class | Disease |
|---|---|
| 0 | COPD (Chronic Obstructive Pulmonary Disease) |
| 1 | Healthy |
| 2 | URTI (Upper Respiratory Tract Infection) |
| 3 | Bronchiectasis |
| 4 | Pneumonia |
| 5 | Bronchiolitis |

---

## Project Structure

```
Lung-Sound-Classification
│
├── extracted_dataset/
├── CNN+LSTM.ipynb
├── final_model.h5
├── lung_disease_predictor.py
├── Images/
└── README.md
```

* `lung_disease_predictor.py` : Streamlit web app
* `final_model.h5` : Trained CNN+LSTM model
* `extracted_dataset/` : Respiratory sound dataset
* `style.css` : Optional custom styles
* `README.md` : Project documentation

---

## Medical Disclaimer

This tool is for educational purposes only. It does not provide medical advice, diagnosis, or treatment. For more information or concerns, please consult a qualified doctor.

---

## Author

Developed by **Naman Saxena**

---

## Contact

For questions or feedback, please reach out via GitHub.