from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Define arrays for greetings and goodbyes
greetings = [
    "hello",
    "hi",
    "hey",
    "howdy",
    "greetings",
    "hola",
    "salutations",
    "yo",
    "what's up",
    "good morning"
]

goodbyes = [
    "goodbye",
    "bye",
    "see you later",
    "see ya",
    "farewell",
    "adios",
    "later",
    "take care",
    "see you soon",
    "so long"
]

# Create training data and corresponding labels
X_train = greetings + goodbyes
y_train = ['greeting'] * len(greetings) + ['goodbye'] * len(goodbyes)

# Define a classification model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Function to predict the category based on numerical input
def predict_category_from_number(number):
    if number >= 0 and number < len(X_train):
        return X_train[number]
    else:
        return "Index out of range"

# Take user input for classification until the user decides to stop
while True:
    user_input = input("Enter a numerical value: ")
    try:
        number = int(user_input)
        output = predict_category_from_number(number)
        print(f"Input {number} is: {output}")
    except ValueError:
        print("Please enter a valid numerical value.")
