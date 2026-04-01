# Tracing Linguistic Evolution with Google Books Ngram Viewer and Time-Aware Word Embeddings

Language is not static — it evolves in response to cultural, political, and technological changes. Words shift in meaning, rise or fall in popularity, and sometimes disappear entirely. By analyzing large text corpora, we can track these linguistic shifts and uncover how historical events have shaped language over time.

In this article, we will:

- Use **Google Books Ngram Viewer** to track word frequency over centuries  
- Analyze the **semantic drift** of words like *democracy* and *liberty*  
- Apply **time-aware word embeddings** to measure how meanings evolve  
- Interpret historical patterns in linguistic change  

By the end, you’ll have a workflow for combining time-series analysis with computational linguistics to study language evolution.

---

## 1. The Role of Time in Linguistic Change

### What Causes Words to Change Over Time?

Several forces influence the way words are used and understood:

- **Political events** – Words like *liberty* and *democracy* surged during revolutions and major geopolitical conflicts.  
- **Technological advances** – *Cloud* meant one thing in 1900 and something very different in 2000.  
- **Cultural shifts** – Terms related to gender, race, and identity have evolved dramatically over the last century.  
- **Media and communication** – New words emerge via social media, while older terms fade or change connotation.

By tracking these changes quantitatively, we gain insight into broader societal transformations and ideological movements.

---

## 2. Tracking “Democracy” and “Liberty” in Google Books Ngram Viewer

[Google Books Ngram Viewer](https://books.google.com/ngrams) analyzes word frequency across millions of digitized books from 1500 to the present. It provides a powerful macro-level view of how often words appear over time.

### Step 1: Querying the Ngram Viewer

We start with a simple query in the web interface:

```text
democracy, liberty  (1800–2019, English, case-insensitive)
```

This returns relative frequencies of *democracy* and *liberty* for each year from 1800 to 2019.

### Step 2: Interpreting the Results

A typical plot shows patterns like:

- A **steady increase** in usage of *democracy* since the late 18th century, coinciding with the **American and French Revolutions**.  
- A **spike in liberty** around the **1860s (American Civil War)** and the **1940s (World War II)**.  
- A **decline in both terms** after the **Cold War**, as political discourse shifts toward concepts like *globalization*, *human rights*, and *governance*.

Ngram Viewer helps us see when concepts enter public consciousness, peak, and fade.

---

## 3. Replicating Ngram Analysis in Python

To go beyond the web interface, we can work programmatically in Python. One option is the `engram` library, which wraps Ngram data access.

```python
import engram
import pandas as pd
import matplotlib.pyplot as plt
```

### Fetching the Data

```python
# Fetch Ngram data for democracy and liberty
ngram_data = engram.get_ngram_data(
    ["democracy", "liberty"],
    start_year=1800,
    end_year=2019,
    corpus="eng_2019",
)
```

Convert the result to a DataFrame:

```python
df = pd.DataFrame(ngram_data)
```

### Plotting Trends

```python
plt.figure(figsize=(10, 5))

for word in df:
    plt.plot(df[word]["year"], df[word]["frequency"], label=word)

plt.xlabel("Year")
plt.ylabel("Relative Frequency")
plt.title("Google Books Ngram Trends for 'democracy' and 'liberty'")
plt.legend()
plt.savefig("ngram_trends.png")
plt.show()
```

### Key Insights

From this type of plot, you typically see:

- **Democracy** gaining prominence in the late **19th and 20th centuries**, peaking during the **Cold War era**.  
- **Liberty** showing **sharp peaks around revolutionary and wartime periods**, including the American Revolution, the Civil War, and major civil rights movements.

Ngram frequency tells us *how often* words are used, but not *how* they are used. For that, we turn to semantics.

---

## 4. Understanding Semantic Drift with Time-Aware Word Embeddings

### What Is Semantic Drift?

**Semantic drift** refers to how the meaning or typical usage of a word changes over time.

For example, for *democracy*:

- In the **18th–19th centuries**, it was associated with **constitutional governance** and the structure of political systems.  
- In the **20th century**, usage was colored by **Cold War ideological conflicts** (e.g., “democratic vs. communist states”).  
- In the **21st century**, *democracy* is often linked to **globalization, elections, human rights, and digital democracy**.

To capture these shifts, we need models that reflect how a word’s *context* changes across eras.

### Using Word Embeddings to Track Meaning Over Time

A **word embedding** is a dense vector representation of a word, learned from its surrounding context in text. Words that appear in similar contexts get similar vectors.

To analyze *democracy* and *liberty* across time, we can:

1. Train **separate embedding models** for different historical periods (e.g., 1800s, 1900s, 2000s).  
2. For each period, inspect **nearest neighbors** of our target words (*democracy*, *liberty*).  
3. Compare how those neighbors change over time to infer semantic drift.

---

## 5. Training Time-Aware Word Embeddings

We’ll illustrate this with **Word2Vec** from `gensim`, using pre-cleaned historical corpora (e.g., subsets of Google Books or other digitized collections).

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
```

### Loading and Tokenizing Historical Text

Assume we have already collected and cleaned representative text for each era:

```python
# Load pre-cleaned historical text corpora
text_1800s = open("books_1800s.txt").read().lower()
text_1900s = open("books_1900s.txt").read().lower()
text_2000s = open("books_2000s.txt").read().lower()
```

Tokenize each corpus:

```python
tokens_1800s = [word_tokenize(text_1800s)]
tokens_1900s = [word_tokenize(text_1900s)]
tokens_2000s = [word_tokenize(text_2000s)]
```

### Training Separate Word2Vec Models

```python
# Train Word2Vec models for different eras
model_1800s = Word2Vec(tokens_1800s, vector_size=100, window=5, min_count=5, sg=0)
model_1900s = Word2Vec(tokens_1900s, vector_size=100, window=5, min_count=5, sg=0)
model_2000s = Word2Vec(tokens_2000s, vector_size=100, window=5, min_count=5, sg=0)
```

Each model now captures word relationships specific to its time slice.

---

## 6. Measuring Semantic Change Over Time

To track **semantic drift**, we look at how the nearest neighbors of a word change across models.

```python
# Get top similar words for "democracy" across eras
print("1800s:", model_1800s.wv.most_similar("democracy", topn=5))
print("1900s:", model_1900s.wv.most_similar("democracy", topn=5))
print("2000s:", model_2000s.wv.most_similar("democracy", topn=5))
```

### Interpreting Results

You might observe patterns like:

- **1800s:** `["republic", "constitution", "monarchy", ...]`  
- **1900s:** `["freedom", "rights", "socialism", ...]`  
- **2000s:** `["governance", "elections", "digital", ...]`  

This suggests that:

- In the **1800s**, *democracy* is clustered with terms describing **forms of government** and political structures.  
- In the **1900s**, it shifts toward **ideological and rights-based** language.  
- In the **2000s**, it appears near terms related to **governance, electoral systems, and digital politics**.

You can repeat the same analysis for *liberty* and compare how its neighbors evolve.

For a more quantitative view, you can also:

- Compute **cosine similarity** between the embeddings of *democracy* across time (e.g., `model_1800s` vs. `model_2000s`).  
- Track how that similarity changes for multiple words to identify which ones are **most semantically unstable** over time.

---

## 7. Limitations and Extensions

Before over-interpreting results, keep in mind:

- **Corpus bias** – Google Books and similar datasets over-represent certain genres, regions, and authors.  
- **OCR and preprocessing noise** – Historical text can be messy, affecting tokenization and embeddings.  
- **Alignment issues** – Models trained separately in different eras live in different vector spaces; more advanced methods align them explicitly.

Possible extensions:

- Apply more sophisticated **diachronic embedding methods** that jointly align vectors across time.  
- Analyze additional concepts such as *freedom, justice, capitalism, feminism* and compare their semantic trajectories.  
- Combine this with **topic modeling** or **sentiment analysis** to understand not just *how often* a word appears, but in what *tone* and *context*.

---

## Conclusion

In this article, we explored:

- **Google Books Ngram Viewer** to track word frequency over centuries  
- **Semantic drift**, the process by which word meanings evolve  
- **Time-aware word embeddings** to quantify and visualize those changes  

Together, these tools give us a powerful framework for studying how language reflects historical, political, and cultural shifts.

You can apply the same methodology to any concept that interests you—*freedom*, *capitalism*, *feminism*, *privacy*, *surveillance*, and more—and start uncovering how our collective vocabulary has evolved over time.
