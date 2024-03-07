from flask import Flask, request, render_template
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle

clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

app = Flask(__name__)
        

stop_words = stopwords.words('english')
def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub(r'[\W+]', ' ', text.lower())

    proter = PorterStemmer()
    text = [proter.stem(word) for word in text.split() if word not in stop_words]

    return " ".join(text)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    if(request.method == 'POST'):
        comment = request.form['text']
        preprocessed_comment = preprocessing(comment)
        comment_list = [preprocessed_comment]
        comment_vector = tfidf.transform(comment_list)
        predction = clf.predict(comment_vector)[0]
        
        return render_template('index.html', prediction = predction)


if __name__ == "__main__":
    app.run(debug=True)