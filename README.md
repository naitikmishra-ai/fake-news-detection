# 🗞️ FNDNet: Intelligent Fake News Detection

![FNDNet Hero Image](https://socialify.git.ci/naitikmishra-ai/fake-news-detection/image?description=1&font=Inter&name=1&pattern=Circuit%20Board&theme=Light) <!-- Placeholder, assuming typical socialify usage -->

> A state-of-the-art Deep Learning implementation based on the Convolutional Neural Network research by Kaliyar et al. (2020) to automatically detect and classify fake news.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](#)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](#)

---

## 📖 Abstract & Objective

With the increasing popularity of social media, the distribution of fake news has become a major threat. Instead of relying on manual features, **FNDNet** is designed to automatically learn discriminatory features for fake news classification through multiple hidden layers in a deep neural network.

This project provides an interactive and intuitive web application to analyze article credibility, train the deep learning model, and evaluate its performance using advanced analytics.

---

## ✨ Features

- **🔍 Live Detection**: Paste any news article text and get an instant credibility analysis, showing whether the article aligns with professional journalism or sensationalized fake news.
- **⚙️ Model Analytics & Training**: Train the deep convolutional layers directly from the UI. Monitor training with interactive convergence graphs, pie charts for dataset distribution, and detailed metrics (Accuracy, Loss, Precision, Recall, F1 Score).
- **🧮 Advanced Evaluation**: View a comprehensive Confusion Matrix of the test data evaluation using Plotly's interactive heatmaps.
- **📚 Research Context**: Learn about the FNDNet architecture and the foundational research paper driving this application.

---

## 🧠 Model Architecture

The neural network is built using TensorFlow and Keras, implementing a robust 1D Convolutional Neural Network (CNN) designed for natural language processing:
- **Embedding Layer**: Converts the input vocabulary into dense vectors.
- **Parallel Convolutional Blocks**: Three parallel `Conv1D` layers with varying kernel sizes (3, 4, 5) to capture different n-gram features, followed by Max Pooling.
- **Deep Convolutional Layers**: Additional `Conv1D` and `MaxPooling1D` layers applied to the concatenated outputs to learn higher-level hierarchical representations.
- **Dense Classifier**: Fully connected layers with Dropout for regularization, outputting a softmax probability over the two classes (Fake vs. Genuine).

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. 

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/naitikmishra-ai/fake-news-detection
   cd fake-news-detection
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, and `tensorflow` installed).*

4. **Prepare the Dataset:**
   Ensure the dataset files `Fake.csv` and `True.csv` are placed inside the `dataset/` directory in the root of the project.

### Running the App

Start the Streamlit application by running:
```bash
streamlit run main.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
fake-news-detection/
├── dataset/
│   ├── Fake.csv        # Dataset containing fake news articles
│   └── True.csv        # Dataset containing genuine news articles
├── main.py             # Main Streamlit application and model architecture
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies (create if not exists)
```

---

## 📊 Dataset

The dataset used for training consists of thousands of categorized news articles:
- **Fake News**: Sensationalized, biased, or verifiably false articles.
- **Genuine News**: Real articles gathered from professional journalism sources (e.g., Reuters).

*Note: The datasets `Fake.csv` and `True.csv` are large and must be processed locally.*

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](#).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Developed with ❤️ using Streamlit & TensorFlow.*
