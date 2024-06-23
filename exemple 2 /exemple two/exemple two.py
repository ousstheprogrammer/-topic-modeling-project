import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


df = pd.DataFrame({'text': ["Sample document one.", "Another sample document.", "Text data for topic modeling."]})


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
translator = str.maketrans('', '', string.punctuation)

def preprocess(text):
   
    text = text.lower()
    
    text = text.translate(translator)
   
    words = word_tokenize(text)
   
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words


df['processed'] = df['text'].apply(preprocess)

print(df['processed'])
