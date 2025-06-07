import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === Fungsi untuk load dan siapkan data ===
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_klasifikasi_mahasiswa_logis_fix.csv")
    X = df.drop(columns=["Status Kelulusan", "Nama"])
    y = df["Status Kelulusan"]

    # Encode fitur kategorikal
    label_encoders = {}
    for column in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    return df, X, y, label_encoders, le_target

# === Load data dan latih model ===
df, X, y, label_encoders, le_target = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# === UI Streamlit ===
st.title("🎓 Prediksi Kelulusan Mahasiswa dengan KNN")
st.write("Masukkan data mahasiswa di bawah ini:")

# Input pengguna
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
status_mahasiswa = st.selectbox("Status Mahasiswa", ["Aktif", "Cuti", "Non-aktif"])
umur = st.slider("Umur", 18, 30, 22)
status_nikah = st.selectbox("Status Nikah", ["Belum Menikah", "Menikah"])
ipk = st.slider("IPK", 2.0, 4.0, 3.0, step=0.01)

# Siapkan data input
data_input = pd.DataFrame([{
    "Jenis Kelamin": jenis_kelamin,
    "Status Mahasiswa": status_mahasiswa,
    "Umur": umur,
    "Status Nikah": status_nikah,
    "IPK": ipk
}])

# Encode input
for col in data_input.select_dtypes(include="object").columns:
    data_input[col] = label_encoders[col].transform(data_input[col])

# Prediksi
prediksi = model.predict(data_input)
hasil = le_target.inverse_transform(prediksi)[0]

# Tampilkan hasil prediksi
st.subheader("📌 Hasil Prediksi:")
st.success(f"Status Kelulusan: {hasil}")

# Tampilkan evaluasi model
st.subheader("📊 Evaluasi Model:")
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
st.write(f"Akurasi Model: *{akurasi:.2f}*")
