# 📰 Deep Learning News Classification

This project applies **Deep Learning and Natural Language Processing (NLP)** techniques to automatically classify news articles into categories.  
It leverages the **News Category Dataset (HuffPost)** and uses the **Universal Sentence Encoder** from TensorFlow Hub for semantic embeddings.

---

## 📌 Project Overview

In today's world, the amount of online news is enormous, and manually categorizing them is inefficient.  
This project aims to build a **news classifier** that predicts the correct category for a given article's headline and short description.

### Key Objectives:
- Preprocess real-world news data  
- Use **Universal Sentence Encoder** embeddings to represent text  
- Train a **deep learning classifier** with TensorFlow/Keras  
- Evaluate performance across multiple categories  

---

## 📊 Dataset

- **Source:** [News Category Dataset v3](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (HuffPost News)  
- **Original Size:** ~200,000 articles across 40+ categories  
- **Filtered Categories for this project:**  
  - 🧪 **SCIENCE**  
  - 🏀 **SPORTS**  
  - 💻 **TECH**  
  - 💼 **BUSINESS**  
  - 🎓 **EDUCATION**  

To ensure **balanced learning**, the dataset was equalized across these categories.

---

## 🧹 Data Preprocessing

1. Dropped irrelevant columns: `link`, `authors`, `date`  
2. Merged `headline` + `short_description` → new column `text`  
3. Encoded categories into numeric labels using **LabelEncoder**  
4. Split dataset into **training (80%)** and **testing (20%)**  

---

## ⚙️ Model Architecture

- **Text Embedding:**  
  - Universal Sentence Encoder (USE) from TensorFlow Hub  
  - Converts raw text into **512-dimensional semantic vectors**  

- **Deep Learning Classifier (Keras):**  
  - Dense layers with dropout  
  - Softmax output for **multi-class classification**  

- **Loss & Optimization:**  
  - Loss: `sparse_categorical_crossentropy`  
  - Optimizer: `Adam`  
  - Metric: `accuracy`  

---

## 🛠️ Tech Stack

- **Language:** Python 3  
- **Libraries:**  
  - TensorFlow / Keras  
  - TensorFlow Hub  
  - Scikit-learn  
  - Pandas, NumPy  

---

## 📂 Project Structure
```

├── DeepLearning\_News\_Classification.ipynb   # Main Jupyter Notebook
├── News\_Category\_Dataset\_v3.json            # Dataset (HuffPost news)
└── README.md                                # Documentation

````

---

## 🚀 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/DeepLearning_News_Classification.git
cd DeepLearning_News_Classification
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Notebook

```bash
jupyter notebook DeepLearning_News_Classification.ipynb
```

---

## 📈 Results

The model was trained and evaluated on the filtered categories.
Performance metrics (accuracy, loss) were measured on the **test set**.

👉 *(Insert your final accuracy, confusion matrix, and plots here once you run the notebook.)*

---

## 🔮 Future Work

* Expand classification to all 40+ categories in the dataset
* Experiment with Transformer models (BERT, DistilBERT, RoBERTa)
* Deploy as a REST API or Streamlit web app
* Apply hyperparameter tuning for better accuracy

---

## 🤝 Contributing

Contributions are welcome!
If you'd like to improve the model, dataset handling, or add deployment options:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature-name`)
5. Create a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute it for educational and research purposes.



