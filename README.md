# Persian Text Similarity & Summarization Toolkit

## 📖 Overview
This project provides a lightweight toolkit for **Persian Natural Language Processing (NLP)**.  
It includes modules for:

- **Text preprocessing** (normalization, tokenization, stopword removal)
- **TF‑IDF calculation** (Term Frequency–Inverse Document Frequency)
- **Cosine similarity** between Persian texts
- **Extractive text summarization** using Parsivar

The goal is to make it easy to compare Persian documents and generate concise summaries.
---

## ⚠️ Important Note on Text Length (Critical)

> **This toolkit is intended ONLY for medium to long Persian texts.**

### ❌ Do NOT use for:

* Single sentences
* Headlines
* Short news snippets
* Tweets or captions

### ✅ Recommended minimum:

* **Similarity**: 2–3 paragraphs per document
* **Summarization**: 10–15 sentences or more

### Why?

TF-IDF and cosine similarity depend on **term distribution statistics**.
Short texts:

* Lack vocabulary diversity
* Produce unstable TF-IDF weights
* Result in misleading similarity scores
* Fail in extractive summarization

---

## ✨ Features

- **Text Preprocessing**
  - Lowercasing and cleaning non‑Persian characters
  - Normalizing Persian digits (`۰۱۲۳...۹`) into Arabic numerals (`0–9`)
  - Tokenization into words and sentences
  - Stopword removal

- **TF‑IDF & Similarity**
  - Compute TF and IDF values across a corpus
  - Generate TF‑IDF vectors
  - Measure cosine similarity between two texts

- **Summarization**
  - Normalize and tokenize text into sentences
  - Extract a percentage of sentences (`ratio`)
  - Limit summary length (`sentence_limit`)
  - Output concise summaries of Persian documents

---

## ⚙️ Installation

### Requirements
- Python 3.x
- Libraries:
  - `math`, `re`, `collections`, `typing`
  - [Parsivar](https://github.com/ICTRC/Parsivar)

### Install Dependencies
```bash
pip install -r requirements.txt
```

Example `requirements.txt`:
```txt
parsivar==0.3
```

---

## 🚀 Usage

### Text Preprocessing
```python
from persian_text_similarity import TextProcessor

text = "این یک متن نمونه است."
cleaned = TextProcessor.preprocess_text(text)
tokens = TextProcessor.tokenize(cleaned)
tokens_no_stopwords = TextProcessor.remove_stopwords(tokens)
print(tokens_no_stopwords)
```

### TF‑IDF & Cosine Similarity
```python
from persian_text_similarity import TFIDFCalculator, SimilarityCalculator

corpus = [tokens_no_stopwords, ["سینما", "هنر", "تاریخ"]]
idf = TFIDFCalculator.calculate_idf(corpus)
tfidf = TFIDFCalculator.calculate_tf_idf(corpus, idf)

similarity = SimilarityCalculator.cosine_similarity(tfidf[0], tfidf[1])
print(f"Cosine similarity: {similarity}")
```

### Text Summarization
```python
from persian_text_summarizer import TextSummarizationPipeline

input_text ="""
سینما یکی از مهم‌ترین پدیده‌های فرهنگی قرن بیستم محسوب می‌شود که
توانسته است مرزهای جغرافیایی و زبانی را درنوردد. این هنر-صنعت با
ترکیب تصویر، صدا، روایت و فناوری، شکل جدیدی از داستان‌گویی را
به جهان معرفی کرده است.

از نخستین نمایش‌های عمومی تصاویر متحرک در اواخر قرن نوزدهم تا
تولید فیلم‌های دیجیتال با جلوه‌های ویژه پیچیده، سینما همواره
در حال تحول بوده است. این تحولات نه‌تنها بر شیوه تولید فیلم،
بلکه بر نحوه دریافت و تفسیر مخاطبان نیز تأثیر گذاشته‌اند.
"""
pipeline = TextSummarizationPipeline(input_text, ratio=0.3, sentence_limit=5)
pipeline.process_and_summarize()
```

---

## 📝 Example

```python
from persian_text_similarity import TextSimilarity

text1 = """
تاریخ سینما به اواخر قرن نوزدهم بازمی‌گردد، زمانی که برادران لومیر
نخستین نمایش عمومی تصاویر متحرک را برگزار کردند. در این دوران،
فیلم‌ها بسیار کوتاه و بدون صدا بودند و بیشتر جنبه سرگرمی داشتند.

با گذشت زمان، فیلم‌سازان به ظرفیت‌های روایی این رسانه پی بردند.
فیلم‌های صامت به داستان‌های پیچیده‌تری پرداختند و کارگردانانی
مانند چارلی چاپلین توانستند احساسات انسانی را بدون دیالوگ منتقل کنند.

ورود صدا به سینما نقطه عطفی در تاریخ این هنر بود. از آن پس،
سینما به یکی از تأثیرگذارترین رسانه‌های فرهنگی و اجتماعی جهان
تبدیل شد و نقش مهمی در شکل‌دهی افکار عمومی ایفا کرد.
"""

text2 = """
سینما از بدو پیدایش خود تاکنون مسیر طولانی و پرفرازونشیبی را طی کرده است.
در ابتدا، فیلم‌ها به صورت صامت و کوتاه تولید می‌شدند و بیشتر جنبه
تفریحی داشتند. اما به‌تدریج، سینما به رسانه‌ای جدی برای روایت
داستان‌های انسانی و اجتماعی تبدیل شد.

پیشرفت فناوری، به‌ویژه ورود صدا و سپس تصویر رنگی، امکانات بیانی
سینما را گسترش داد. در دهه‌های اخیر، جلوه‌های ویژه دیجیتال و
فناوری‌های نوین، سینما را وارد مرحله‌ای تازه کرده‌اند.

امروزه سینما نه‌تنها یک صنعت بزرگ اقتصادی است، بلکه یکی از مهم‌ترین
ابزارهای فرهنگی در جهان معاصر به شمار می‌رود.
"""


similarity_calc = TextSimilarity(text1, text2)
similarity = similarity_calc.process_and_calculate_similarity()

print(f"Cosine similarity between the two texts: {similarity}")
```

---

## 📂 File Structure
```
your-project/
├── LICENSE
├── persian_text_similarity.py   # Preprocessing, TF-IDF, similarity
├── persian_text_summarizer.py   # Summarization logic
├── README.md                    # Documentation
├── requirements.txt             # Dependencies
```

---

## 🤝 Contributing
Contributions are welcome!  
1. Fork the repository  
2. Create a new branch for your feature or fix  
3. Add tests to cover your changes  
4. Submit a pull request  

---

## 📜 License
This project is open‑source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements
- [Parsivar](https://github.com/ICTRC/Parsivar) for Persian NLP tools  
- Standard TF‑IDF and cosine similarity methods from information retrieval
