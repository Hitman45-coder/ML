from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "Cats and dogs are pets."
]

count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(documents)

print(X_count.toarray())
print(count_vectorizer.get_feature_names_out())
