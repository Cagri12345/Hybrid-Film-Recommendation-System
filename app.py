import streamlit as st
import pandas as pd
from recommender import load_data, prepare_user_movie_df, hybrid_recommendation

st.title("🎬 Hybrid Film Öneri Sistemi")

# Veri dosyaları
movie_path = r"C:\Users\cagri\PycharmProjects\PythonProject1\Hybrid_Recommender\movie.csv"
rating_path = r"C:\Users\cagri\PycharmProjects\PythonProject1\Hybrid_Recommender\rating.csv"

# Veriyi yükle
movie_df, rating_df = load_data(movie_path, rating_path)
user_movie_df = prepare_user_movie_df(movie_df, rating_df)

# Film seçimi
st.subheader("🎞️ Film Seçin")
movie_list = user_movie_df.columns.tolist()
selected_movie = st.selectbox("Bir film seçin", movie_list)

# Kullanıcı ID girişi
target_user_id = st.number_input("🎯 Kullanıcı ID'si girin", min_value=1, value=108170)

if st.button("🔍 Hibrit Önerileri Göster"):
    with st.spinner("Öneriler hazırlanıyor..."):
        hybrid_df = hybrid_recommendation(user_movie_df, movie_df, rating_df, movie_df, selected_movie, target_user_id)
    st.subheader("✅ Önerilen Filmler:")
    for i, row in hybrid_df.iterrows():
        st.write(f"**{row['title']}** — Skor: {round(row['hybrid_score'], 3)}")
