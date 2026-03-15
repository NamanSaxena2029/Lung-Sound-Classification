import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from tensorflow.keras.models import load_model
import os
import gdown

st.set_page_config(page_title="Respiratory Disease Prediction System", page_icon="🫁", layout="wide")

# -------------------------
# MODEL LOAD (CACHED)
# -------------------------
@st.cache_resource
def get_model():

    model_path = "final_model.h5"

    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1KIor01gadBWov5ctUyqszWLDornZgcHs"
        gdown.download(url, model_path, quiet=False)

    model = load_model(model_path, compile=False)

    return model


model = get_model()

# Sidebar Navigation
with st.sidebar:
    st.title("🎵 Model Information")
    st.markdown("**Model:** CNN + LSTM Hybrid")
    st.markdown(
        """
        <div style='font-size: 15px;'>
        <b>About the Model:</b><br>
        This web app uses a hybrid deep learning model combining 1D Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) layers. The CNN layers extract important features from lung sound audio, while the LSTM layer captures temporal patterns and dependencies. This architecture is trained on a large dataset of respiratory sounds and can classify diseases such as COPD, URTI, Bronchiectasis, Pneumonia, Bronchiolitis, and Healthy cases. The model uses advanced feature extraction and data augmentation for robust predictions.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("**Supported Audio Formats**")
    st.markdown("- WAV files (.wav)\n- MP3 files (.mp3)\n- M4A files (.m4a)")
    st.markdown("---")
    st.markdown("**How to Use**")
    st.markdown("1. Upload an audio file\n2. Wait for processing\n3. Review the prediction results\n4. Consult healthcare professionals")
    st.markdown("---")
    st.markdown("**Feature Extraction**")
    st.markdown("- MFCC coefficients\n- Chroma features\n- Mel spectrogram\n- Spectral contrast\n- Tonnetz features")
    st.markdown("---")
    st.markdown("**Recommendations**")
    st.info("This tool is for educational purposes only. For medical advice, consult a qualified doctor.")

# Header
st.markdown(
    """
    <div style="text-align:left;">
        <h1 style="color:#2E8B57;">🫁 Respiratory Disease Prediction System</h1>
        <h4 style="color:#555;">Upload a lung sound file to get an AI-powered analysis</h4>
    </div>
    """, unsafe_allow_html=True
)

# Medical Disclaimer
st.warning("Medical Disclaimer - Please Read")

st.success("CNN Model loaded successfully!")

index_to_label = {0: "COPD", 1: "Healthy", 2: "URTI", 3: "Bronchiectasis", 4: "Pneumonia", 5: "Bronchiolitis"}

disease_descriptions = {
    "COPD": "Chronic Obstructive Pulmonary Disease affecting airflow.",
    "Healthy": "No signs of respiratory disease detected.",
    "URTI": "Upper Respiratory Tract Infection affecting nose, throat, and airways.",
    "Bronchiectasis": "Chronic condition with abnormal widening of airways.",
    "Pneumonia": "Infection that inflames air sacs in one or both lungs.",
    "Bronchiolitis": "Inflammation of the small airways in the lung."
}

disease_tips = {
    "COPD": "Avoid smoking, practice breathing exercises, follow prescribed medications.",
    "Healthy": "Maintain a balanced diet, exercise regularly, avoid respiratory irritants.",
    "URTI": "Stay hydrated, rest well, use steam inhalation, consult a doctor if symptoms persist.",
    "Bronchiectasis": "Perform chest physiotherapy, avoid infections, take prescribed antibiotics.",
    "Pneumonia": "Complete prescribed antibiotics, rest and hydrate, seek medical attention for severe symptoms.",
    "Bronchiolitis": "Monitor breathing, keep airways clear, consult a pediatrician for children."
}

def audio_features(filename):

    sound, sample_rate = librosa.load(filename)

    stft = np.abs(librosa.stft(sound))

    mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40), axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate), axis=1)

    mel = np.mean(librosa.feature.melspectrogram(y=sound, sr=sample_rate), axis=1)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate), axis=1)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sample_rate), axis=1)

    concat = np.concatenate((mfccs, chroma, mel, contrast, tonnetz))

    return concat, sound, sample_rate


# Upload Section
st.markdown("## 📁 Upload Audio Sample")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:

    st.info(f"Filename: {uploaded_file.name}\nFile size: {uploaded_file.size/1024:.1f} KB")

    temp_path = "temp_uploaded.wav"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(temp_path, format="audio/wav")

    try:

        with st.spinner("Analyzing lung sound..."):

            features, sound, sr = audio_features(temp_path)

            features = np.reshape(features, (1, features.shape[0], 1))

            prediction = model.predict(features, verbose=0)

        predicted_class = np.argmax(prediction)

        predicted_label = index_to_label[predicted_class]

        confidence = float(prediction[0][predicted_class])

        st.markdown("### 🩺 Analysis Results")

        st.markdown(
            f"""
            <div style='background:#f9f9f9;padding:20px;border-radius:10px;'>
                <h3 style='color:#2E8B57;'>Primary Prediction: <span style='color:#2E8B57;'>{predicted_label}</span></h3>
                <h4>Confidence Score: <span style='color:#2E8B57;'>{confidence:.2%}</span></h4>
                <p style='color:#555;background:#eef;border-radius:5px;padding:8px;'>About {predicted_label}: {disease_descriptions[predicted_label]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Confidence Distribution")

        prob_dict = {index_to_label[i]: float(prediction[0][i]) for i in range(len(index_to_label))}

        st.bar_chart(prob_dict)

        st.markdown("#### Detailed Probability Scores")

        st.table({
            "Disease": list(prob_dict.keys()),
            "Probability (%)": [f"{v*100:.2f}%" for v in prob_dict.values()],
            "Confidence": ["High" if i == predicted_class else "Low" for i in range(len(index_to_label))]
        })

        st.markdown("#### Recommendations")

        st.info(f"Potential Concerns Detected: {predicted_label}\n\n{disease_tips[predicted_label]}")

    except Exception as e:

        st.error(f"Error processing file: {e}")

    os.remove(temp_path)
