import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
data = pd.read_csv("tech_stack_dataset.csv")

print("Dataset:")
print(data)

# Convert Skills into TF-IDF Vectors
tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(data["Skills"])

print("\nTF-IDF Matrix Shape:")
print(tfidf_matrix.shape)

# Calculate Cosine Similarity
similarity = cosine_similarity(tfidf_matrix)

print("\nSimilarity Matrix:")
print(similarity)

# Take User Input
user_skills = input("\nEnter your skills: ")

# Convert User Skills into TF-IDF Vector
user_vector = tfidf.transform([user_skills])

# Compare User Skills with Dataset
similarity_scores = cosine_similarity(user_vector, tfidf_matrix)

# Find Best Match
best_match = similarity_scores.argmax()

# Display Recommendation
print("\nRecommended Role:", data.iloc[best_match]["Role"])