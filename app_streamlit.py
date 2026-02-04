import pandas as pd
import joblib
import streamlit as st

st.sidebar.title("Machine Learning")
st.sidebar.success("Dibuat ole Almira Zahra")

st.set_page_config(
	page_title="🍅 regressi tomat",
	page_icon=":alien:"
)



st.title(" 🍅 Regressi Penjualan Tomat")
st.markdown("Aplikasi machine learning regression untuk menghitung total penjualan tomat berdasarkan fitur `Harga, Hari, Cuaca, dan Promo`")

model_rf = joblib.load("model_rf.joblib")

harga = st.slider("Harga", 0, 20000, 7000)
cuaca = st.selectbox("Cuaca",["Cerah","Berawan","Mendung","Hujan"])
hari = st.selectbox("Hari",["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"])
promo = st.pills("Promo", ["Ya","Tidak"],default="Tidak")

if st.button("Prediksi"):
	data_baru = pd.DataFrame([[harga,cuaca,hari,promo]], columns=["Harga","Cuaca","Hari","Promo"])
	prediksi = model_rf.predict(data_baru)[0]
	st.success(f"Model memprediksi total penjualannya **{prediksi:.0f}**")
	st.balloons()