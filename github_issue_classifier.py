import numpy as np
import joblib
import requests
import contractions
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')

MODEL_PATH = "best_model/nn_embeddings_tuned.joblib"
WORD2VEC_MODEL_PATH = "word2vec_model/word2vec_model.joblib"

def fetch_github_issue(owner, repo, issue_number):
    response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}")
    if response.status_code == 200:
        issue = response.json()
        title = issue['title']
        body = issue['body']
        
        return title, body
    else:
        print("Failed to fetch issue from GitHub.")
        return None, None

def clean_text(title, body):
    title = title.replace("\r", " ")
    title = title.replace("\n", " ")
    title = title.replace('"', '')
    title = title.lower()

    body = body.replace("\r", " ")
    body = body.replace("\n", " ")
    body = body.replace('"', '')
    body = body.lower()

    contractions.fix(title)
    contractions.fix(body)
    
    punctuation_signs = [r"\?", r"\:", r"\!", r"\.", r"\,", r"\;"]

    for punct_sign in punctuation_signs:
        title = re.sub(punct_sign, ' ', title)
        body = re.sub(punct_sign, ' ', body)
        
    title = re.sub("'s", "", title)
    body = re.sub("'s", "", body)
    
    return title, body

def lemmatize(title, body):
    wordnet_lemmatizer = WordNetLemmatizer()

    title_lemmatized_list = []
    body_lemmatized_list = []
    
    title_text_words = title.split(" ")
    body_text_words = body.split(" ")

    for word in title_text_words:
        title_lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

    for word in body_text_words:
        body_lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    title_lemmatized_text = " ".join(title_lemmatized_list)
    body_lemmatized_text = " ".join(body_lemmatized_list)
    
    return title_lemmatized_text, body_lemmatized_text

def remove_stopwords(title, body):
    stop_words = list(stopwords.words('english'))

    for stop_word in stop_words:
        regex_stopword = r"\b" + re.escape(stop_word) + r"\b"
        title = re.sub(regex_stopword, '', title)
        body = re.sub(regex_stopword, '', body)
    
    title = re.sub(r'\s+', ' ', title)
    body = re.sub(r'\s+', ' ', body)

    return title, body

def preprocess_data(title, body):
    title, body = clean_text(title, body)
    title, body = lemmatize(title, body)
    title, body = remove_stopwords(title, body)
    
    return title, body

def get_average_word2vec(combined_text, model):
    words = combined_text.split()
    feature_vector = np.zeros((100,), dtype="float32")
    num_words = 0

    for word in words:
        if word in model.wv:
            num_words += 1
            feature_vector = np.add(feature_vector, model.wv[word])

    if num_words:
        feature_vector = np.divide(feature_vector, num_words)
        
    return feature_vector

def create_embedding(title, body, model):
    combined_text = title + " " + body
    return get_average_word2vec(combined_text, model)

if __name__ == "__main__":
    
    nn_model = joblib.load(MODEL_PATH)
    word2vec_model = joblib.load(WORD2VEC_MODEL_PATH)
    
    while True:
        print("\nChoose an option:")
        print("a. Input issue title and body directly")
        print("b. Fetch issue from GitHub")
        print("c. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == 'a':
            title = input("Enter the issue title: ")
            body = input("Enter the issue body: ")
            
        elif choice == 'b':
            owner = input("Enter the GitHub repository owner's name: ")
            repo = input("Enter the GitHub repository name: ")
            issue_number = input("Enter the GitHub issue number: ")
            
            title, body = fetch_github_issue(owner, repo, issue_number)
            if title is None and body is None:
                continue
            
            print(f"Your title: {title}, Body: {body}")
            
        elif choice == 'c':
            print("Exiting the program.")
            break
        
        else:
            print("Invalid input, please choose again.")
            continue
        
        processed_title, processed_body = preprocess_data(title, body)
        embedding = create_embedding(processed_title, processed_body, word2vec_model)
        
        prediction = nn_model.predict(embedding.reshape(1, -1))[0]
        
        prediction_dict = {0: "Bug", 1: "Feature", 2: "Question"}
        printed_prediction = prediction_dict.get(prediction, "Unknown")
        
        print("..........................................................")
        
        print(f"The predicted class for the issue is: {printed_prediction}")
