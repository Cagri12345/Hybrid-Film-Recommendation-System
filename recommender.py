import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Film ve puan verilerini yüklemek için bir fonksiyon tanımlıyoruz
def load_data(movie_path, rating_path):
    """
       Verilen dosya yollarından film ve kullanıcı puanı verilerini yükleyen fonksiyon.

       Args:
            movie_path: str - Filmlerin bulunduğu CSV dosyasının yolu
            rating_path: str - Kullanıcıların filmlere verdiği puanların bulunduğu CSV dosyasının yolu

        Returns:
            movie: DataFrame - Film bilgilerini içeren pandas veri çerçevesi
            rating: DataFrame - Puan bilgilerini içeren pandas veri çerçevesi
    """
    # Film verilerini pandas ile okuyoruz
    movie = pd.read_csv(movie_path)

    # Kullanıcı puanlarını pandas ile okuyoruz
    rating = pd.read_csv(rating_path)

    return movie, rating




def prepare_user_movie_df(movie, rating, min_rating_count=1000):
    """
        Kullanıcı-film derecelendirme matrisi hazırlar.

        Args:
            movie (DataFrame): Filmlere ait metadata bilgileri.
            rating (DataFrame): Kullanıcıların filmlere verdiği puanlar.
            min_rating_count (int): Bir filmin kullanıcı-film matrisine alınabilmesi için
                                    en az kaç kez oylanmış olması gerektiğini belirtir.

        Returns:
            user_movie_df (DataFrame): Kullanıcıları satırlarda, filmleri sütunlarda gösteren
                                       ve her hücrede kullanıcının filme verdiği puanı içeren pivot tablo.
    """
    # Puanlama verisi ile film bilgileri birleştirilir.
    rating_with_movies = rating.merge(movie, how="left", on="movieId")

    # Her filmin kaç kez değerlendirildiği hesaplanır.
    comment_counts = rating_with_movies["title"].value_counts()

    # Yetersiz sayıda oylanan filmler belirlenir (rare_movies)
    rare_movies = comment_counts[comment_counts < min_rating_count].index

    # Yetersiz oylanan filmler veri setinden çıkarılır.
    rating_filtered = rating_with_movies[~rating_with_movies["title"].isin(rare_movies)]

    # Kullanıcı-film derecelendirme matrisi oluşturulur.
    user_movie_df = rating_filtered.pivot_table(index="userId", columns="title", values="rating")

    return user_movie_df



def user_based_recommendations(user_movie_df, target_user_id, top_n=5):
    """
    User-Based öneri sistemi: Hedef kullanıcıya benzer kullanıcıları bulur.

    Args:
        user_movie_df (DataFrame): Kullanıcı-film derecelendirme matrisi.
        target_user_id (int): Öneri yapılacak hedef kullanıcının ID'si.
        top_n (int): Kaç tane en benzer kullanıcı döndürüleceği (şu an kullanılmıyor, istenirse eklenebilir).

    Returns:
        top_users_df (DataFrame): Hedef kullanıcıya benzer kullanıcıların ID'leri ve korelasyonları.
    """

    # Hedef kullanıcının verisi alınır.
    random_user_df = user_movie_df.loc[[target_user_id]]

    # Hedef kullanıcının izlediği filmler listelenir.
    movies_watched = user_movie_df.loc[target_user_id].dropna().index.tolist()

    # Sadece hedef kullanıcının izlediği filmleri içeren yeni bir DataFrame oluşturulur.
    movies_watched_df = user_movie_df[movies_watched]

    # Diğer kullanıcıların, hedef kullanıcının izlediği kaç filmi izlediği hesaplanır.
    user_movie_count = movies_watched_df.notnull().sum(axis=1).reset_index()
    user_movie_count.columns = ["userId", "movie_count"]

    # Hedef kullanıcının izlediği film sayısı alınır.
    movie_count = len(movies_watched)

    # Hedef kullanıcının izlediği filmlerin %60'ından fazlasını izlemiş kullanıcılar seçilir.
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > movie_count * 0.6]["userId"].tolist()

    # Bu benzer kullanıcıların sadece ortak filmler üzerindeki puanları alınır.
    similar_users_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]

    # Korelasyon matrisi oluşturulur (kullanıcılar arası benzerlik).
    corr_df = similar_users_df.T.corr()

    # Hedef kullanıcı ile diğer kullanıcılar arasındaki korelasyonlar alınır.
    user_corr = corr_df[target_user_id]

    # Hedef kullanıcıyla 0.55 üzeri korelasyona sahip kullanıcılar seçilir (kendisi hariç).
    top_users = user_corr[(user_corr > 0.55) & (user_corr.index != target_user_id)]

    # Sonuçlar DataFrame’e dönüştürülür.
    top_users_df = top_users.reset_index()
    top_users_df.columns = ['userId', 'correlation']

    return top_users_df


def generate_user_based_scores(top_users_df, rating):
    # Kullanıcı benzerlik skorları ile rating verisini birleştir
    top_users_ratings = top_users_df.merge(rating, on='userId')

    # Her kullanıcının film puanını kendi benzerlik oranı (correlation) ile çarp
    top_users_ratings["weighted_rating"] = top_users_ratings["correlation"] * top_users_ratings["rating"]

    # Her film için ağırlıklı ortalama puanı hesapla
    recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"}).reset_index()

    # Yalnızca ortalama ağırlıklı puanı 3.0'dan büyük olan filmleri filtrele
    recommendation_df = recommendation_df[recommendation_df["weighted_rating"] > 3.0]

    # Önerileri ağırlıklı puana göre azalan şekilde sırala
    recommendation_df = recommendation_df.sort_values("weighted_rating", ascending=False)

    return recommendation_df



def item_based_recommendations(user_movie_df, movie_title, top_n=5):
    """
    Belirli bir film temel alınarak, benzer filmleri korelasyon değerlerine göre önerir.

    Args:
    user_movie_df (DataFrame): Kullanıcı-film puanlama matrisi
    movie_title (str): Öneri yapılacak referans filmin adı
    top_n (int): Kaç adet benzer film önerileceği (varsayılan 5)

    Returns:
    recommendation_df (DataFrame): En yüksek korelasyona sahip benzer filmleri içeren tablo
    """
    # Hedef filmin kullanıcı puanlarını al
    target_movie_ratings = user_movie_df[movie_title]

    # Hedef film ile diğer tüm filmler arasındaki korelasyonları hesapla
    correlation_df = user_movie_df.corrwith(target_movie_ratings)

    # Korelasyonları DataFrame'e çevir ve eksik değerleri kaldır
    correlation_df = correlation_df.dropna().to_frame(name="correlation")

    # Korelasyonlara göre azalan şekilde sırala
    correlation_df = correlation_df.sort_values("correlation", ascending=False)

    # Hedef film hariç en yüksek korelasyona sahip filmleri al
    recommendation_df = correlation_df[correlation_df.index != movie_title].head(top_n)

    return recommendation_df



def hybrid_recommendation(user_movie_df, movie, rating, movie_df, selected_movie_title, target_user_id, top_n=10):
    """
    Kullanıcı-temelli ve öğe-temelli öneri sistemlerinin skorlarını birleştirerek hibrit öneriler oluşturur.

    Args:
        user_movie_df (DataFrame): Kullanıcı-film puanlama matrisi
        movie (DataFrame): Film verisi (movie.csv)
        rating (DataFrame): Kullanıcı puanları verisi (rating.csv)
        movie_df (DataFrame): Film verisi (movie.csv), film ID ve isim için
        selected_movie_title (str): Öneri yapılacak referans film adı (item-based için)
        target_user_id (int): Öneri yapılacak kullanıcı ID'si (user-based için)
        top_n (int): Kaç tane öneri döndürüleceği (varsayılan 10)

    Returns:
        hybrid_df (DataFrame): Kullanıcı-temelli ve öğe-temelli skorların ağırlıklı birleşimi ile oluşan öneri listesi
    """
    # Kullanıcı-temelli öneriler için benzer kullanıcılar ve skorlar
    top_users_df = user_based_recommendations(user_movie_df, target_user_id)
    user_scores_df = generate_user_based_scores(top_users_df, rating)
    user_scores_df = user_scores_df.merge(movie_df[['movieId', 'title']], on='movieId')

    # Kullanıcı-temelli skorları düzenle
    user_df = user_scores_df[["title", "weighted_rating"]].copy()
    user_df.columns = ["title", "user_score"]

    # Öğeye dayalı öneriler
    item_df = item_based_recommendations(user_movie_df, selected_movie_title).reset_index()
    item_df.columns = ["title", "item_score"]

    # Skorları 0-1 aralığında normalize et
    user_df["user_score"] = (user_df["user_score"] - user_df["user_score"].min()) / (user_df["user_score"].max() - user_df["user_score"].min())
    item_df["item_score"] = (item_df["item_score"] - item_df["item_score"].min()) / (item_df["item_score"].max() - item_df["item_score"].min())

    # İki skor setini birleştir, eksik değerleri 0 ile doldur
    hybrid_df = pd.merge(user_df, item_df, on="title", how="outer").fillna(0)

    # Ağırlıklı skor hesapla (%60 user-based, %40 item-based)
    hybrid_df["hybrid_score"] = 0.6 * hybrid_df["user_score"] + 0.4 * hybrid_df["item_score"]

    # Skora göre sırala ve en iyi önerileri seç
    hybrid_df = hybrid_df.sort_values("hybrid_score", ascending=False).head(top_n)

    return hybrid_df

