# Tracing Linguistic Evolution with Google Books Ngram Viewer and Time-Aware Word Embeddings

⚠️ Data Requirements: Requires Google Books Ngram data or historical text corpora

## Introduction

Language is not static — it evolves in response to cultural, political, and technological changes. Words shift in meaning, rise or fall in popularity, and sometimes disappear entirely. By analyzing vast textual datasets, we can track these linguistic shifts and uncover how historical events have shaped language over time.

## The Role of Time in Linguistic Change

### What Causes Words to Change Over Time?

- Political Events: Words like *liberty* and *democracy* surged during revolutions
- Technological Advances: *Cloud* meant one thing in 1900, something else in 2000
- Cultural Shifts: Terms related to gender, race, identity evolved dramatically
- Media and Communication: New words emerge through social media, older terms fade

## Tracking "Democracy" and "Liberty" in Google Books Ngram Viewer

Google Books Ngram Viewer analyzes word frequency across millions of digitized books from 1500 to the present.

### Step 1: Querying the Ngram Viewer

Visit https://books.google.com/ngrams and search:
```
democracy, liberty (1800-2019, English, case-insensitive)
```

### Step 2: Interpreting the Results

The output shows:
- Steady increase in *democracy* since late 18th century (American and French Revolutions)
- Spike in liberty around 1860s (American Civil War) and 1940s (World War II)
- Decline after Cold War as discourse shifted toward *globalization* and *human rights*

### Step 3: Replicating Ngram Analysis in Python

```python
import engram  # pip install engram
import pandas as pd
import matplotlib.pyplot as plt

# Fetch Ngram data
ngram_data = engram.get_ngram_data(
    ["democracy", "liberty"], 
    start_year=1800, 
    end_year=2019, 
    corpus="eng_2019"
)

# Convert to DataFrame
df = pd.DataFrame(ngram_data)

# Plot trends
plt.figure(figsize=(10, 5))
for word in df:
    plt.plot(df[word]['year'], df[word]['frequency'], label=word)

plt.xlabel("Year")
plt.ylabel("Relative Frequency")
plt.title("Google Books Ngram Trends for 'Democracy' and 'Liberty'")
plt.legend()
plt.savefig("ngram_trends.png")
plt.show()
```

⚠️ Note: The `engram` library is unofficial. For production, use Google Ngram API or download datasets from https://storage.googleapis.com/books/ngrams/books/datasetsv3.html

## Understanding Semantic Drift with Time-Aware Word Embeddings

### What is Semantic Drift?

Semantic drift refers to how the meaning of words changes over time:
- *Democracy* in 18th century: constitutional governance
- *Democracy* in 20th century: Cold War ideological conflicts
- *Democracy* in 21st century: globalization, elections, digital democracy

### Using Word Embeddings to Track Meaning

A word embedding is a mathematical representation of a word based on its context. By training embeddings on historical texts from different time periods, we can see how meaning has changed.

### Training Time-Aware Word Embeddings

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Load historical text datasets (pre-cleaned)
text_1800s = open("books_1800s.txt").read().lower()
text_1900s = open("books_1900s.txt").read().lower()
text_2000s = open("books_2000s.txt").read().lower()

# Tokenize
tokens_1800s = [word_tokenize(text_1800s)]
tokens_1900s = [word_tokenize(text_1900s)]
tokens_2000s = [word_tokenize(text_2000s)]

# Train Word2Vec models for different eras
model_1800s = Word2Vec(tokens_1800s, vector_size=100, window=5, min_count=5, sg=0)
model_1900s = Word2Vec(tokens_1900s, vector_size=100, window=5, min_count=5, sg=0)
model_2000s = Word2Vec(tokens_2000s, vector_size=100, window=5, min_count=5, sg=0)
```

### Measuring Semantic Change Over Time

```python
# Get top similar words for democracy
print("1800s:", model_1800s.wv.most_similar("democracy", topn=5))
print("1900s:", model_1900s.wv.most_similar("democracy", topn=5))
print("2000s:", model_2000s.wv.most_similar("democracy", topn=5))
```

Example Output (Hypothetical):
- 1800s: ["republic", "constitution", "monarchy"]
- 1900s: ["freedom", "rights", "socialism"]
- 2000s: ["governance", "elections", "digital"]

This shows how *democracy* evolved from a governmental system to a rights-based ideology to a technological/digital term.

## Data Requirements for Production

⚠️ This analysis requires substantial historical text data:

### Data Sources
- Google Books Ngram Corpus: https://storage.googleapis.com/books/ngrams/books/datasetsv3.html
- COHA (Corpus of Historical American English): https://www.english-corpora.org/coha/
- Project Gutenberg: https://www.gutenberg.org/
- Internet Archive: https://archive.org/details/texts

### Processing Requirements
- Storage: 100GB+ for full Google Ngram corpus
- Compute: Word2Vec training requires significant CPU/GPU
- Memory: 16GB+ RAM for large corpora

## Applications

### Historical Linguistics
Track language evolution across centuries

### Cultural Analysis
Understand how societal changes affect language

### Computational Humanities
Quantify linguistic trends in literature

### Political Science
Analyze rhetoric changes over time

## Key Insights

### Democracy and Liberty
- Democracy gained prominence in late 19th century, peaked during Cold War
- Liberty had sharp peak during American Revolution and civil rights movements

### Semantic Drift
- Words shift meaning in response to historical events
- Temporal embeddings capture these changes mathematically

## Limitations

### Data Bias
- Google Books corpus overrepresents English-language academic texts
- Not representative of spoken language or informal writing
- Selection bias toward published materials

### Computational Challenges
- Training embeddings requires substantial compute
- Alignment across time periods is non-trivial
- Statistical significance testing is complex

### Interpretation
- Correlation ≠ causation for linguistic changes
- Cultural context crucial for understanding shifts

## Conclusion

In this chapter, we explored:
- ✓ Google Books Ngram Viewer to track word frequency over centuries
- ✓ Semantic drift to understand how meanings change
- ✓ Time-aware word embeddings to model linguistic evolution

This methodology can be applied to other words like *freedom*, *justice*, *capitalism*, *feminism*, etc.

## Next Steps

- Apply forecasting techniques to predict cultural/linguistic trends
- Expand to multilingual analysis
- Integrate with social media data for modern language evolution

## Resources

- Google Books Ngram Viewer: https://books.google.com/ngrams
- Gensim Documentation: https://radimrehurek.com/gensim/
- COHA Corpus: https://www.english-corpora.org/coha/
- Historical Text Processing: https://programminghistorian.org/

---

Note: This article demonstrates concepts. Production analysis requires substantial computational resources and historical text corpora.
