# Hybrid Film Recommendation System 🎬

## About the Project
This project is a **Hybrid Film Recommendation System** that combines user-based and item-based recommendation algorithms. It generates personalized movie recommendations using movie data and users' rating information.

The system merges the scores of two different recommendation approaches (user similarity and item similarity) with weighted averages, providing balanced and more personalized suggestions.

## Features
- **User-Based Filtering:** Recommends movies liked by similar users.
- **Item-Based Filtering:** Suggests movies similar to the selected one.
- **Hybrid Model:** Combines scores from both methods with weighted averaging.
- **Streamlit Interface:** Easy-to-use interactive web app for the recommendation system.

## Technologies Used
- Python 3.x
- Pandas
- Scikit-learn (for Cosine Similarity)
- Streamlit (for web interface)

## File Structure
- `recommender.py`: Contains the recommendation algorithm functions.
- `app.py`: Streamlit-based user interface.
- `movie.csv`: Dataset containing movie information.
- `rating.csv`: Dataset containing user ratings.

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Hybrid-Film-Recommendation-System.git
