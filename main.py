import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Préparation des données
nltk.download('punkt_tab')
questions = ["Salut", "Comment ça va?", "Quel est ton nom?", "Que fais-tu?", "Au revoir"]
reponses = ["Salut!", "Je vais bien, merci!", "Je suis un chatbot IA.", "Je suis ici pour discuter.", "Au revoir!"]

# Vectorisation et entraînement du modèle
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
model = make_pipeline(vectorizer, SVC(kernel='linear'))

# Entraîner le modèle
model.fit(questions, reponses)

# Fonction pour répondre à une question
def chatbot(question):
    return model.predict([question])[0]

# Interaction avec l'utilisateur
print("Tape 'exit' pour quitter")
while True:
    user_input = input("Vous: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Au revoir!")
        break
    response = chatbot(user_input)
    print(f"Chatbot: {response}")