import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Concatenate, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

MAX_LEN = 500 
EMBEDDING_DIM = 100 
VOCAB_SIZE = 20000

def build_fndnet_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN):
    inputs = Input(shape=(max_len,))
    
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len)(inputs)
    
    conv_3 = Conv1D(filters=128, kernel_size=3, activation='relu')(embedding_layer)
    conv_4 = Conv1D(filters=128, kernel_size=4, activation='relu')(embedding_layer)
    conv_5 = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding_layer)
    
    pool_3 = MaxPooling1D(pool_size=5)(conv_3)
    pool_4 = MaxPooling1D(pool_size=5)(conv_4)
    pool_5 = MaxPooling1D(pool_size=5)(conv_5)
    
    concat = Concatenate(axis=1)([pool_3, pool_4, pool_5])
    
    conv_deep_1 = Conv1D(filters=128, kernel_size=5, activation='relu')(concat)
    pool_deep_1 = MaxPooling1D(pool_size=5)(conv_deep_1)
    
    conv_deep_2 = Conv1D(filters=128, kernel_size=5, activation='relu')(pool_deep_1)
    pool_deep_2 = MaxPooling1D(pool_size=5)(conv_deep_2)
    
    flat = Flatten()(pool_deep_2)
    dense_1 = Dense(128, activation='relu')(flat)
    
    dropout = Dropout(0.5)(dense_1) 
    
    output = Dense(2, activation='softmax')(dropout)
    model = Model(inputs=inputs, outputs=output)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

@st.cache_data 
def load_and_process_data():
    if not os.path.exists("dataset/Fake.csv") or not os.path.exists("dataset/True.csv"):
        return None, None, None, None, None, None, None

    df_fake = pd.read_csv("dataset/Fake.csv")
    df_true = pd.read_csv("dataset/True.csv")
    
    df_true['text'] = df_true['text'].astype(str).str.replace(r'^.*?\([A-Za-z]+\)\s*-\s*', '', regex=True)
    df_true['text'] = df_true['text'].str.lower()
    df_fake['text'] = df_fake['text'].astype(str).str.lower()
    
    df_fake['label'] = 0 
    df_true['label'] = 1 
    
    fake_count = len(df_fake)
    true_count = len(df_true)
    
    df = pd.concat([df_fake, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    texts = df['text'].astype(str).tolist()
    labels = df['label'].values
    labels_cat = to_categorical(labels, num_classes=2)
    
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    X = pad_sequences(sequences, maxlen=MAX_LEN)
    X_train, X_test, y_train, y_test = train_test_split(X, labels_cat, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, tokenizer, fake_count, true_count

st.set_page_config(page_title="FNDNet Analytics", layout="wide", page_icon="🗞️")

st.title("🗞️ FNDNet: Intelligent Fake News Detection")
st.markdown("A deep learning implementation based on the Convolutional Neural Network research by Kaliyar et al. (2020).")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Live Detection", "⚙️ Model Analytics & Training", "📚 Research Context"])

if 'model' not in st.session_state:
    st.session_state.model = build_fndnet_model()
    st.session_state.tokenizer = None

with tab1:
    st.header("Analyze Article Credibility")
    news_input = st.text_area("Paste the news article text here:", height=200)
    
    if st.button("Run FNDNet Analysis", type="primary"):
        if st.session_state.tokenizer is None:
            st.error("⚠️ Model is not trained yet. Please go to the 'Model Analytics' tab and run the pipeline.")
        elif not news_input:
            st.warning("Please enter text to analyze.")
        else:
            with st.spinner("Analyzing linguistic patterns..."):
                clean_input = news_input.lower()
                sequence = st.session_state.tokenizer.texts_to_sequences([clean_input])
                padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
                
                prediction = st.session_state.model.predict(padded_sequence)
                fake_prob = prediction[0][0] * 100
                true_prob = prediction[0][1] * 100
                
                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)
                
                if true_prob > fake_prob:
                    col1.metric("Classification", "✅ Genuine News", f"{true_prob:.2f}% Confidence")
                    st.success("The network architecture indicates this article aligns with the semantic patterns of professional journalism.")
                else:
                    col1.metric("Classification", "🚨 Fake News", f"-{fake_prob:.2f}% Confidence")
                    st.error("The network architecture detected sensationalized language or bias consistent with unreliable sources.")
                
                with col2:
                    st.markdown("**Confidence Breakdown**")
                    st.progress(int(true_prob), text=f"Genuine ({true_prob:.1f}%)")
                    st.progress(int(fake_prob), text=f"Fake ({fake_prob:.1f}%)")

with tab2:
    st.header("Model Analytics & Performance")
    
    if st.button("▶️ Initialize Training Pipeline"):
        with st.spinner("Loading and preprocessing datasets..."):
            X_train, X_test, y_train, y_test, tokenizer, fake_count, true_count = load_and_process_data()
            
        if X_train is not None:
            st.session_state.tokenizer = tokenizer
            
            st.subheader("📊 Dataset Distribution")
            fig_pie = px.pie(
                values=[fake_count, true_count], 
                names=['Fake News', 'Genuine News'], 
                color_discrete_sequence=['#ef553b', '#00cc96'],
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            with st.spinner("Training deep convolutional layers (Monitoring validation loss)..."):
                early_stopper = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
                history = st.session_state.model.fit(
                    X_train, y_train, 
                    epochs=5, 
                    batch_size=128, 
                    validation_split=0.1,
                    callbacks=[early_stopper]
                )
                
            st.success("🎉 Training Complete! Analyzing final metrics on test data...")
            
            final_acc = history.history['accuracy'][-1] * 100
            final_val_acc = history.history['val_accuracy'][-1] * 100
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            st.divider()
            st.subheader("🎯 Final Model Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Training Accuracy", f"{final_acc:.2f}%")
            m2.metric("Validation Accuracy", f"{final_val_acc:.2f}%")
            m3.metric("Training Loss", f"{final_loss:.4f}")
            m4.metric("Validation Loss", f"{final_val_loss:.4f}")
            st.divider()
            
            epochs_range = range(1, len(history.history['accuracy']) + 1)
            df_acc = pd.DataFrame({'Training': history.history['accuracy'], 'Validation': history.history['val_accuracy']}, index=epochs_range)
            df_loss = pd.DataFrame({'Training': history.history['loss'], 'Validation': history.history['val_loss']}, index=epochs_range)
            
            st.subheader("📈 Convergence Graphs")
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("**Accuracy over Epochs**")
                st.line_chart(df_acc)
            with col_chart2:
                st.markdown("**Loss over Epochs**")
                st.line_chart(df_loss)
                
            st.subheader("🧮 Confusion Matrix (Test Data Evaluation)")
            with st.spinner("Evaluating test dataset predictions..."):
                y_pred = st.session_state.model.predict(X_test)
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_test, axis=1)
                
                cm = confusion_matrix(y_true_classes, y_pred_classes)

                precision = precision_score(y_true_classes, y_pred_classes, zero_division=0)
                recall = recall_score(y_true_classes, y_pred_classes, zero_division=0)
                f1 = f1_score(y_true_classes, y_pred_classes, zero_division=0)

                st.subheader("📌 Precision / Recall / F1 Score")
                p1, p2, p3 = st.columns(3)
                p1.metric("Precision", f"{precision:.4f}")
                p2.metric("Recall", f"{recall:.4f}")
                p3.metric("F1 Score", f"{f1:.4f}")
                
                fig_cm = px.imshow(
                    cm, 
                    text_auto=True, 
                    aspect="auto", 
                    labels=dict(x="Predicted Label", y="True Label", color="Count"), 
                    x=['Fake', 'Genuine'], 
                    y=['Fake', 'Genuine'], 
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
        else:
            st.error("Datasets not found. Please ensure `Fake.csv` and `True.csv` are in the `dataset` folder.")

with tab3:
    st.header("Research Context: FNDNet")
    st.markdown("""
    This project implements the state-of-the-art **FNDNet (Fake News Detection Network)** architecture, a deep convolutional neural network proposed in the 2020 study by researchers Kaliyar, Goswami, Narang, and Sinha.
    """)
    st.subheader("Abstract & Objective")
    st.write("With the increasing popularity of social media, the distribution of fake news has become a major threat. Instead of relying on manual features, FNDNet is designed to automatically learn discriminatory features for fake news classification through multiple hidden layers in a deep neural network.")