import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# --- 1. KONFIGURASI HALAMAN & TEMA SPOTIFY ---
st.set_page_config(page_title="Spotify Hit Predictor", page_icon="🎧", layout="centered")

# Inject Custom CSS agar mirip Spotify
st.markdown("""
<style>
    /* Background utama gelap ala Spotify */
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    /* Sidebar hitam pekat */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    /* Warna teks hijau Spotify untuk judul */
    h1, h2, h3 {
        color: #1DB954 !important;
        font-family: 'Circular', sans-serif;
    }
    /* Tombol hijau ala Spotify */
    .stButton>button {
        background-color: #1DB954;
        color: black;
        border-radius: 50px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1ed760;
        color: white;
    }
    /* Menyesuaikan warna teks di slider agar tetap terbaca */
    label {
        color: #B3B3B3 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
model = joblib.load('spotify_hit_model.pkl')

st.title("🎧 Spotify Hit Predictor")
st.write("Prediksi apakah sebuah lagu punya potensi jadi hits (Popular) atau flop berdasarkan racikan audionya!")

# --- 3. INPUT SIDEBAR (UI Bahasa Manusia) ---
st.sidebar.header("Atur Vibe Lagu Kamu 🎛️")
acousticness = st.sidebar.slider("🎸 Nuansa Akustik", 0.0, 1.0, 0.5)
danceability = st.sidebar.slider("💃 Asik Buat Joget?", 0.0, 1.0, 0.7)
energy = st.sidebar.slider("⚡ Energi Lagu", 0.0, 1.0, 0.6)
instrumentalness = st.sidebar.slider("🎹 Fokus Instrumen", 0.0, 1.0, 0.0)
liveness = st.sidebar.slider("🎤 Vibe Konser Live", 0.0, 1.0, 0.1)
loudness = st.sidebar.slider("🔊 Tingkat Kebisingan (dB)", -60.0, 0.0, -5.0)
speechiness = st.sidebar.slider("🗣️ Banyak Kata-katanya?", 0.0, 1.0, 0.05)
tempo = st.sidebar.slider("⏱️ Tempo (BPM)", 50.0, 200.0, 120.0)
valence = st.sidebar.slider("😊 Vibe Emosi (Happy/Positif)", 0.0, 1.0, 0.5)
duration_ms = st.sidebar.number_input("⏳ Durasi Lagu (ms)", min_value=30000, max_value=600000, value=200000)

# --- 4. EKSEKUSI PREDIKSI ---
if st.button("Prediksi Potensi Lagu!"):
    # Syarat Latency < 100 ms
    start_time = time.time()
    
    input_data = pd.DataFrame([{
        'acousticness': acousticness, 'danceability': danceability, 'duration_ms': duration_ms,
        'energy': energy, 'instrumentalness': instrumentalness, 'key': 5,
        'liveness': liveness, 'loudness': loudness, 'mode': 1,
        'speechiness': speechiness, 'tempo': tempo, 'time_signature': 4, 'valence': valence
    }])
    
    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = 0
            
    input_data = input_data[expected_cols]
    
    # Paksa 2D array dengan nama kolom (bebas dari error merah)
    input_2d = np.array(input_data).reshape(1, -1)
    input_df = pd.DataFrame(input_2d, columns=expected_cols)
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    end_time = time.time()
    latency = (end_time - start_time) * 1000

    st.write("---")
    if prediction == 1:
        st.success(f"🔥 PREDIKSI: LAGU HITS! (Probabilitas: {probability[1]*100:.2f}%)")
    else:
        st.error(f"📉 PREDIKSI: FLOP / Kurang Populer. (Probabilitas: {probability[0]*100:.2f}%)")
        
        # Fitur Sistem Pendukung Keputusan (Actionable Feedback)
        st.warning("💡 **Saran AI buat Produser biar lagunya jadi Hits:**")
        saran = []
        if danceability < 0.5:
            saran.append("- Beat-nya kurang asik. Coba bikin lebih *danceable* (naikin Danceability).")
        if energy < 0.5:
            saran.append("- Lagunya terlalu lemes. Tambahin distorsi atau bass biar lebih nendang (naikin Energy).")
        if loudness < -10.0:
            saran.append("- Mixing-nya terlalu pelan. Coba *mastering* ulang biar lebih kencang (naikin Loudness).")
        if tempo < 90.0:
            saran.append("- Temponya agak lambat, rawan bikin ngantuk. Coba cepetin dikit BPM-nya.")
            
        if len(saran) > 0:
            for s in saran:
                st.write(s)
        else:
            st.write("- Racikan fiturnya udah unik, tapi mungkin pasarnya segmented. Coba eksperimen di genre lain!")
    
    st.caption(f"⏱️ Model Execution Latency: {latency:.2f} ms")