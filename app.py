from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load and preprocess data
def load_and_preprocess_data():
    # Load datasets
    books = pd.read_csv("Books.csv", dtype={'ISBN': str}, low_memory=False)
    ratings = pd.read_csv("Ratings.csv", dtype={'User-ID': int, 'ISBN': str, 'Book-Rating': int})
    users = pd.read_csv("Users.csv", dtype={'User-ID': int}, low_memory=False)

    # Drop missing titles and filter relevant ratings
    books.dropna(subset=['Book-Title'], inplace=True)
    ratings = ratings[ratings['Book-Rating'] > 0]

    # Merge datasets
    merged_df = ratings.merge(books, how="left", on="ISBN")
    return merged_df

# Recommendation logic
def recommend_books(user_id, data, top_n=5):
    # Filter data for user-specific books
    user_books = data[data['User-ID'] == user_id]
    if user_books.empty:
        return []

    # Find books rated by similar users
    similar_users = data[data['ISBN'].isin(user_books['ISBN']) & (data['User-ID'] != user_id)]
    recommended_books = (
        similar_users.groupby('Book-Title')['Book-Rating']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    # Return book titles and average ratings
    recommendations = recommended_books.reset_index()
    recommendations.columns = ['Book Title', 'Average Rating']
    return recommendations.to_dict(orient='records')

# Preprocess data once during app startup
data = load_and_preprocess_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = int(request.form.get('user_id'))
        recommendations = recommend_books(user_id, data, top_n=5)
        if not recommendations:
            error = "No recommendations found. Try a different User ID."
            return render_template('index.html', error=error)
        return render_template('index.html', recommendations=recommendations)
    except ValueError:
        return render_template('index.html', error="Invalid User ID. Please enter a valid number.")

if __name__ == '__main__':
    app.run(debug=True)
