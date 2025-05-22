# NLP-With-Disaster-Tweets
Used RNN and BERT-based models to identify tweets denoting natural disasters

Here is how they performed (test set accuracy) - 

  Base Model:
    Logistic Regression with TF-IDF Vectorization: ~57% accuracy
    SVM with TF-IDF Vectorization: ~58% accuracy
  
  RNN-Based Models (using GloVe 300d embeddings):
    Bidirectional LSTM layer + Simple LSTM layer with single dense layer: ~81.3% accuracy 
    Two Bidirectional LSTM layers with 2 dense layers: ~81.5% accuracy
    
  Transformer-Based Models (using GloVe 300d embeddings):
    Pre-trained BERT with 12 layers and self-attention: ~82.2% accuracy
    DistilBERT with 6 layers and self-attention: ~82.6% accuracy

Learnings:
  1. Choice of model:
     - Wider LSTM structure with Batch Normalization gave us strong accuracy (0.815)
     - Transformer-based models (BERT, DistilBERT) performed better overall, but not significantly
      better (0.826)
     - LSTM required much more fine-tuning and modifications to the structure than BERT or DistilBERT
  
  3. Text Normalization: Careful choices around pre-processing improved model performance: We chose lemmatization to retain as much information from the word as possible

  4. Vectorization: Plays a key role in improving model performance
     - TF-IDF vectorization did not capture dimensionality well
     - Experimenting with methods like GloVE which capture semantic relationships helped us
     - Here too we experimented with 50d and 300d embeddings
     - Use of 300d embeddings improved validation accuracy by 2-3%, using same model
       architectures
