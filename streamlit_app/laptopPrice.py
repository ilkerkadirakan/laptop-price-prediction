import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from streamlit import columns
import requests
from bs4 import BeautifulSoup

# Model ve veri yükleme
model_path = 'streamlit_app/model.pkl'
data_path = 'streamlit_app/dataset.pkl'

try:
    model = joblib.load(model_path)
    st.success("Model başarıyla yüklendi")
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

# Kullanıcı arayüzü
st.title("Laptop Özellikleri Seçimi")

brand = st.selectbox("Marka", ['ASUS', 'Lenovo', 'acer', 'Avita', 'HP', 'DELL', 'MSI', 'APPLE'])
processor_brand = st.selectbox("İşlemci Markası", ['Intel', 'AMD', 'M1'])
processor_name = st.selectbox("İşlemci Adı",
                              ['Core i3', 'Core i5', 'Celeron Dual', 'Ryzen 5', 'Core i7', 'Core i9', 'M1',
                               'Pentium Quad', 'Ryzen 3', 'Ryzen 7', 'Ryzen 9'])
processor_gnrtn = st.selectbox("İşlemci Nesli", ['10th', 'Not Available', '11th', '7th', '8th', '9th', '4th', '12th'])

ram_gb = st.number_input("RAM (GB)", min_value=0, step=1)
ram_type = st.selectbox("RAM Türü", ['DDR4', 'LPDDR4', 'LPDDR4X', 'DDR5', 'DDR3', 'LPDDR3'])

ssd = st.number_input("SSD (GB)", min_value=0, step=1)
hdd = st.number_input("HDD (GB)", min_value=0, step=1)

os = st.selectbox("İşletim Sistemi", ['Windows', 'DOS', 'Mac'])
os_bit = st.radio("İşletim Sistemi Bit Değeri", ['64-bit', '32-bit'])

graphic_card_gb = st.number_input("Ekran Kartı (GB)", min_value=0, step=1)
weight = st.selectbox("Ağırlık Kategorisi", ['Casual', 'ThinNlight', 'Gaming'])
warranty = st.number_input("Garanti Süresi (Yıl)", min_value=0.0, step=0.5)

touchscreen = st.radio("Dokunmatik Ekran", ['No', 'Yes'])
msoffice = st.radio("MS Office Yüklü mü?", ['No', 'Yes'])

rating = st.slider("Puan", 1.0, 5.0, step=0.5)
num_ratings = st.number_input("Puan Sayısı", min_value=0, step=1)
num_reviews = st.number_input("Yorum Sayısı", min_value=0, step=1)

if st.button("Tahmin Yap"):
    input_data = pd.DataFrame([{
        "brand": brand,
        "processor_brand": processor_brand,
        "processor_name": processor_name,
        "processor_gnrtn": processor_gnrtn,
        "ram_gb": ram_gb,
        "ram_type": ram_type,
        "ssd": ssd,
        "hdd": hdd,
        "os": os,
        "os_bit": os_bit,
        "graphic_card_gb": graphic_card_gb,
        "weight": weight,
        "warranty": warranty,
        "touchscreen": touchscreen,
        "msoffice": msoffice,
        "rating": rating,
        "num_ratings": num_ratings,
        "num_reviews": num_reviews
    }])

    # Preprocessing

    numerical_columns = input_data.select_dtypes(include=['int64', 'float64']).columns

    def handle_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.clip(df[column], lower_bound, upper_bound)
        return df[column]

    for col in numerical_columns:
        input_data[col] = handle_outliers(input_data, col)

    input_data[numerical_columns] = np.log1p(input_data[numerical_columns])

    input_data["processor_gnrtn"] = input_data["processor_gnrtn"].str.extract('(\\d+)').astype(float)

    def assign_generation(row):
        if pd.isna(row["processor_gnrtn"]):
            if "Ryzen 3" in row["processor_name"]:
                return 10
            elif "Ryzen 5" in row["processor_name"]:
                return 10
            elif "Ryzen 7" in row["processor_name"]:
                return 11
            elif "Ryzen 9" in row["processor_name"]:
                return 11
            elif "Celeron Dual" in row["processor_name"]:
                return 8
            elif "Pentium Quad" in row["processor_name"]:
                return 8
            elif "Core i5" in row["processor_name"]:
                return 10
            else:
                return 10
        return row["processor_gnrtn"]

    input_data["processor_gnrtn"] = input_data.apply(assign_generation, axis=1)

    input_data["processor_gnrtn"] = np.log1p(input_data["processor_gnrtn"])

    input_data["processor_gnrtn"] = input_data.apply(assign_generation, axis=1)
    input_data["processor_gnrtn"] = np.log1p(input_data["processor_gnrtn"])

    input_data["proccessor_gen_ram_interaction"] = np.expm1(input_data["processor_gnrtn"]) * np.expm1(input_data["ram_gb"])
    input_data["proccessor_gen_ram_interaction"] = np.log1p(input_data["proccessor_gen_ram_interaction"])

    input_data['processor_gpu_interaction'] = np.expm1(input_data['processor_gnrtn']) * np.expm1(
        input_data['graphic_card_gb'])
    input_data['processor_gpu_interaction'] = np.log1p(input_data['processor_gpu_interaction'])

    input_data['processor_ssd_interaction'] = np.expm1(input_data['processor_gnrtn']) * np.expm1(input_data['ssd'])
    input_data['processor_ssd_interaction'] = np.log1p(input_data['processor_ssd_interaction'])

    input_data['ssd_hdd_ratio'] = np.expm1(input_data['ssd']) / (np.expm1(input_data['hdd']) + 1e-5)
    input_data['ssd_hdd_ratio'] = np.log1p(input_data['ssd_hdd_ratio'])


    one_hot_encoded_cols = ["brand", "processor_name", "ram_type", "processor_brand", "weight"]
    for col in one_hot_encoded_cols:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_data = encoder.fit_transform(input_data[[col]])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]))
        input_data = pd.concat([input_data, encoded_df], axis=1)
        input_data = input_data.drop(columns=[col])

    label_encoded_cols = ["os_bit", "touchscreen", "msoffice", "os"]
    for col in label_encoded_cols:
        encoder = LabelEncoder()
        input_data[col] = encoder.fit_transform(input_data[col])

    try:
        dataset = joblib.load(data_path)
        dataset = dataset.drop(columns=['Price'])
        required_columns = dataset.columns
    except Exception as e:
        st.error("Veri yüklenirken bir hata oluştu")
        st.stop()

    input_data = input_data.reindex(columns=required_columns, fill_value=0)

    missing_cols = set(required_columns) - set(input_data.columns)
    extra_cols = set(input_data.columns) - set(required_columns)

    #st.write(f"Modelin beklediği fakat eksik olan sütunlar: {missing_cols}")
    #st.write(f"Fazladan bulunan sütunlar: {extra_cols}")

    prediction_price = model.predict(input_data)[0]

    def get_exchange_rate(base_currency, target_currency):
        url = f"https://www.x-rates.com/calculator/?from={base_currency}&to={target_currency}&amount=1"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        rate = None
        try:
            rate = soup.find("span", class_="ccOutputTrail").previous_sibling.get_text()
            return float(rate.replace(',', ''))
        except Exception as e:
            print(f"Hata oluştu: {e}")
            return None

    rate_TRY = get_exchange_rate("INR", "TRY")
    rate_USD = get_exchange_rate("INR", "USD")

    if rate_TRY and rate_USD:
        converted_price_TRY = rate_TRY * predicted_price
        converted_price_USD = rate_USD * predicted_price
        print(f"Tahmini Fiyat (TRY): {converted_price_TRY:,.2f} TL")
        print(f"Tahmini Fiyat (USD): {converted_price_USD:,.2f} USD")
    else:
        print("Döviz kuru verisi alınamadı.")

    st.write(f"Tahmini Fiyat: {predicted_price:,.2f} Rupi")
    st.write(f"Tahmini Fiyat: {converted_price_TRY:,.2f} TL")
    st.write(f"Tahmini Fiyat: {converted_price_USD:,.2F} Dolar")





