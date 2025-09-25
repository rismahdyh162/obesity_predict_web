import pickle

with open('model_obesitas.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Cek tipe objek yang dimuat
print(type(loaded_model))

# Contoh prediksi dengan model yang dimuat
sample_data = [[25, 2, 3, 3, 4]]  # Ganti dengan data uji sample
prediction = loaded_model.predict(sample_data)
print("Hasil prediksi:", prediction)