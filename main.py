import requests
import nltk
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math



# URL of the online text
url = 'https://raw.githubusercontent.com/teropa/nlp/master/resources/corpora/gutenberg/shakespeare-hamlet.txt'

# Sending a GET request to the URL
response = requests.get(url)

# Checking if the request was successful (status code 200)
if response.status_code == 200:
    corpus = response.text
    print("I successfully retrieved Hamlet.")
else:
    print(f"Failed to retrieve content, status code: {response.status_code}")


class Document:
    def __init__(self, lines):
        self.lines = lines

    def tokenize(self) -> list[str]:
        tokenized_text = self.lines.lower()
        nospecial_tokenized_text = re.sub(r'[^a-z ]+', '', tokenized_text)
        clean_tokenized_text = nospecial_tokenized_text.split(" ")
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in clean_tokenized_text]
        return stemmed_words


doc = Document(corpus)
tokens = doc.tokenize()
print(tokens[:100])


X = tokens
y = tokens


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#####

class Corpus:
    def __init__(self, url):
        self.url = url
        self.text = None
        self.tokenized_text = None
        self.train_text = None
        self.test_text = None
        self.vocabulary = None
        self.ngram_counts = None
        self.ngram_probabilities = None

    def load(self):
        print("Loading corpus from URL:", self.url)
        response = requests.get(self.url)
        if response.status_code == 200:
            print("Corpus loaded successfully.")
            self.text = response.text
        else:
            print(f"Failed to retrieve content, status code: {response.status_code}")

    def tokenize(self):
        print("Tokenizing corpus...")
        self.tokenized_text = nltk.word_tokenize(self.text)

    def split(self, train_ratio=0.8):
        print("Splitting corpus into training and testing chunks...")
        split_index = int(train_ratio * len(self.tokenized_text))
        self.train_text = self.tokenized_text[:split_index]
        self.test_text = self.tokenized_text[split_index:]

    def build_vocabulary(self):
        print("Building vocabulary of tokenized training corpus...")
        self.vocabulary = sorted(set(self.train_text))
        print("Vocabulary built.")

    def write_vocabulary_to_file(self, filename):
        print("Writing vocabulary to file:", filename)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("\n".join(self.vocabulary))
        print("Vocabulary written to file.")

    def count_ngrams(self, n):
        print(f"Counting {n}-grams over tokenized training corpus...")
        self.ngram_counts = defaultdict(int)
        for i in range(len(self.train_text) - n + 1):
            ngram = tuple(self.train_text[i:i+n])
            self.ngram_counts[ngram] += 1
        print(f"{n}-grams counted successfully.")

    def write_ngram_counts_to_file(self, filename):
        print("Writing n-gram counts to file:", filename)
        with open(filename, 'w', encoding='utf-8') as file:
            for ngram, count in self.ngram_counts.items():
                file.write(' '.join(ngram) + ': ' + str(count) + '\n')
        print("N-gram counts written to file.")

    def calculate_ngram_probabilities(self):
        print("Calculating n-gram probabilities...")
        self.ngram_probabilities = {}
        prefix_counts = defaultdict(int)
        for ngram, count in self.ngram_counts.items():
            prefix = ngram[:-1]
            prefix_counts[prefix] += count
        for ngram, count in self.ngram_counts.items():
            prefix = ngram[:-1]
            self.ngram_probabilities[ngram] = count / prefix_counts[prefix]
        print("N-gram probabilities calculated.")
        return self.ngram_probabilities

    def calculate_perplexity(self, n):
        print("Calculating perplexity over tokenized test corpus...")
        log_sum = 0
        N = len(self.test_text)
        for i in range(len(self.test_text) - n + 1):
            ngram = tuple(self.test_text[i:i+n])
            log_sum += -math.log2(self.ngram_probabilities.get(ngram, 1e-10))
        perplexity = 2 ** (log_sum / N)
        print("Perplexity calculated.")
        return perplexity

# Create Corpus instance
corpus = Corpus('https://raw.githubusercontent.com/teropa/nlp/master/resources/corpora/gutenberg/shakespeare-hamlet.txt')

# Step 1: Load corpus
corpus.load()

# Step 2: Tokenize loaded corpus
corpus.tokenize()

# Step 3: Split corpus into training and testing chunks
corpus.split()

# Step 4: Obtain vocabulary of tokenized training corpus
corpus.build_vocabulary()

# Step 5: Write vocabulary to file
corpus.write_vocabulary_to_file("vocabulary1.txt")

# Step 6: Count n-grams over tokenized training corpus
corpus.count_ngrams(2)  # Change to 3 for 3-grams

# Step 7: Write n-gram counts to a file
corpus.write_ngram_counts_to_file("ngram_counts1.txt")

# Step 8: Calculate n-gram probabilities
sample_probability = corpus.calculate_ngram_probabilities()
first_10_ngrams = list(sample_probability.items())[:10]
for ngram, probability in first_10_ngrams:
    print(f"{ngram}: {probability}")

# Step 9: Calculate perplexity over tokenized test corpus
perplexity = corpus.calculate_perplexity(2)  # Change to 3 for 3-grams
print("Perplexity:", perplexity)

