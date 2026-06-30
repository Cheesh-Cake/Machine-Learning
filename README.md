# 🎧 Spotify Hit Predictor
**Classification-Based Decision Support System**

Proyek akhir untuk mata kuliah Machine Learning (COMP6577001). 
Aplikasi ini memprediksi apakah sebuah lagu berpotensi menjadi "Hit" atau "Flop" di pasaran berdasarkan fitur audionya menggunakan algoritma *Random Forest Classifier*.

## 👨‍💻 Developer
* **Randy Adika Fathoni** - BINUS University
* **Achmad Sanusi** - BINUS University

## ✨ Fitur Utama
* **Prediksi Instan:** Mengklasifikasikan lagu secara *real-time* (latensi < 100ms) melalui antarmuka Streamlit.
* **Saran Produser (Actionable Feedback):** Memberikan rekomendasi penyesuaian elemen audio (seperti *tempo*, *energy*, atau *danceability*) jika lagu diprediksi kurang populer.
* **Akurasi Tinggi:** Model dilatih menggunakan dataset *Spotify Audio Features 2019*.

## 🛠️ Teknologi yang Digunakan
* Python
* Scikit-Learn (Random Forest)
* Pandas & NumPy
* Streamlit (Deployment)

## 🚀 Cara Menjalankan Secara Lokal
1. Clone repository ini.
2. Install semua kebutuhan library: `pip install -r requirements.txt`
3. Jalankan aplikasi web: `streamlit run app.py`
   
