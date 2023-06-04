import fasttext
import re
import string


def preprocess(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def analyze_sentiment(model, text):
    # Preprocess the text
    preprocessed_text = preprocess(text)

    # Perform sentiment analysis using the FastText model
    prediction = model.predict(preprocessed_text)
    sentiment = prediction[0][0].split('__')[-1]

    return sentiment


def main():
    # Load the FastText model
    model = fasttext.load_model('model_amzn.bin')

    # Get user input and provide sentiment analysis results
    while True:
        text = input("Enter text (or 'quit' to exit): ")

        if text == 'quit':
            break

        sentiment = analyze_sentiment(model, text)
        print("Sentiment:", sentiment)


if __name__ == '__main__':
    main()
