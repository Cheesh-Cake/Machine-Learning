import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Data
df = pd.read_csv('SpotifyAudioFeaturesApril2019.csv')

# 2. Preprocessing & ML Problem Formulation
# Kita buat target variabel baru: 'Hit' (1) jika popularity > 50, dan 'Flop' (0) jika di bawahnya.
df['is_hit'] = (df['popularity'] > 50).astype(int)

# Hapus kolom yang bukan fitur numerik atau tidak relevan untuk prediksi
features_to_drop = ['artist_name', 'track_name', 'track_id', 'popularity'] 
X = df.drop(columns=features_to_drop, errors='ignore')
y = df['is_hit']

# Tangani jika ada missing values (opsional tapi disarankan)
X = X.fillna(X.median())

# 3. Split & Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluasi
y_pred = model.predict(X_test)
print("--- HASIL EVALUASI UNTUK PPT ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5. Save Model
joblib.dump(model, 'spotify_hit_model.pkl')
print("Model berhasil disimpan sebagai 'spotify_hit_model.pkl'")