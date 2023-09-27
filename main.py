import nltk
from nltk.corpus import inaugural
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download NLTK resources if needed
nltk.download('inaugural', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load the inaugural corpus
corpus = ' '.join(inaugural.words())  # Combine all words in the inaugural corpus

# Tokenize the text into words
tokens = word_tokenize(corpus)

# Remove punctuation and convert to lowercase
words = [word.lower() for word in tokens if word.isalnum()]

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

# Word frequency analysis
word_freq = Counter(filtered_words)

# Create a simple bar chart for the top N words (optional)
N = 10  # You can change this to get more or fewer words
top_words = word_freq.most_common(N)

# Extract words and frequencies for the chart (optional)
words, frequencies = zip(*top_words)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Bar chart
ax1.bar(words, frequencies)
ax1.set_xlabel('Words')
ax1.set_ylabel('Frequency')
ax1.set_title(f'Top {N} Words')
ax1.tick_params(axis='x', rotation=45)

# Subplot 2: Word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
ax2.imshow(wordcloud, interpolation='bilinear')
ax2.axis('off')
ax2.set_title('Word Cloud')

# Adjust layout and display
plt.tight_layout()
plt.show()

