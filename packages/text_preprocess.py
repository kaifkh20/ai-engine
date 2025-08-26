import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords',quiet=True)
nltk.download('punkt_tab',quiet=True)



def tokenize(s):
    
    stop_words = set(stopwords.words('english'))
    # Tokenize and normalize to lowercase
    tokens = word_tokenize(s.lower())
    # Filter out stopwords
    return [w for w in tokens if w not in stop_words and w.isalnum()]

def stemming(tokens):
    stemmer = PorterStemmer()

    return [stemmer.stem(t) for t in tokens]

def text_preprocess(s):
    print(f"Text Pre-processing ... Done")
    tokens = tokenize(s)

    stemmerized_tokens = stemming(tokens)

    return stemmerized_tokens
        
