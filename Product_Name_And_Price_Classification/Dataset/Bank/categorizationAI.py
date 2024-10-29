import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pickle


similar_categories = [
    [1, "Supermarkets and Groceries", "Inventory Purchases"],
    [2, "Healthcare", "Insurance (Health)"],
    [3, "Internal Account Transfer", "Banking Fees"],
    [4, "Payroll", "Payroll"],
    [5, "Third Party", "Professional Fees"],
    [6, "Department Stores", "Store Supplies"],
    [7, "Insurance", "Insurance (Liability)"],
    [8, "Loans", "Loan Repayments"],
    [9, "Convenience Stores", "Store Supplies"],
    [10, "Gas Stations", "Fuel Costs"],
    [11, "Transfer Debit", "Banking Fees"],
    [12, "Travel", "Travel Expenses"],
    [13, "Clothing and Accessories", "Uniforms"],
    [14, "Shops", "Store Fixtures"],
    [15, "Utilities", "Utilities"],
    [16, "Check Deposit", "Banking Fees"],
    [17, "Food and Beverage Services", "Staff Meals"],
    [18, "ATM", "ATM Fees"],
    [19, "Telecommunication Services", "Telephone Service"],
    [20, "Transfer Deposit", "Banking Fees"],
    [21, "Payment", "POS System Fees"],
    [22, "Tax Refund", "Taxes (Sales)"],
    [23, "Gyms and Fitness Centers", "Employee Wellness Programs"],
    [24, "Interest", "Interest Expenses"],
    [25, "Digital Entertainment", "Music Licensing Fees"],
    [26, "Transfer Credit", "Banking Fees"],
    [27, "Service", "Professional Fees"],
    [28, "Bank Fees", "Banking Fees"],
    [29, "Arts and Entertainment", "Promotional Materials"],
    [30, "Restaurants", "Restaurants"]
]


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Print column names for verification
    print("Column names in CSV file:", df.columns.tolist())
    
    # Check the first few rows to understand the data
    print("First few rows of the DataFrame:")
    print(df.head())
    
    # Extract data from columns
    inputs = df['TXN_DESC'].tolist()
    outputs = df['category'].astype(str).tolist()
    
    return inputs, outputs

def tokenize_and_encode(texts, max_length, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    return padded_sequences, tokenizer

def create_seq2seq_model(input_vocab_size, output_vocab_size, input_seq_length, output_seq_length, embedding_dim=256, lstm_units=512):
    # Encoder
    encoder_inputs = Input(shape=(input_seq_length,))
    encoder_embedding = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)(encoder_embedding)
    encoder_output, forward_h, forward_c = encoder_lstm  # Unpacking three values: output, hidden state, and cell state
    
    # Decoder
    decoder_inputs = Input(shape=(output_seq_length,))
    decoder_embedding = Embedding(output_vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True)(decoder_embedding, initial_state=[forward_h, forward_c])
    
    # Attention
    attention = Attention()([decoder_lstm, encoder_output])
    concat = Concatenate(axis=-1)([decoder_lstm, attention])
    
    # Dense layer
    output = Dense(output_vocab_size, activation='softmax')(concat)
    
    model = Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, epochs=10, batch_size=64):
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2
    )

def predict_sequence(model, input_seq, max_output_length, tokenizer, decoder_target_data):
    # Pad the input sequence to the required length
    input_seq = pad_sequences([input_seq], maxlen=max_input_length, padding='post')
    
    # Initialize the decoder input with zeros
    decoder_input = np.zeros((1, max_output_length))
    
    # Get the predictions
    prediction = model.predict([input_seq, decoder_input])
    
    # Get the index of the most probable token at each time step
    predicted_indices = np.argmax(prediction, axis=-1)[0]
    
    # Convert the indices back to text
    predicted_text = tokenizer.sequences_to_texts([predicted_indices])[0]
    
    return predicted_text

def save_model_and_metadata(model, model_file, encoder_tokenizer, decoder_tokenizer, max_input_length, max_output_length):
    # Save model
    model.save(model_file)
    
    # Save tokenizers and max lengths
    with open('encoder_tokenizer_category.pkl', 'wb') as f:
        pickle.dump(encoder_tokenizer, f)
    with open('decoder_tokenizer_category.pkl', 'wb') as f:
        pickle.dump(decoder_tokenizer, f)
    with open('max_lengths_category.pkl', 'wb') as f:
        pickle.dump({'max_input_length': max_input_length, 'max_output_length': max_output_length}, f)
    
    print(f"Model and metadata saved to {model_file}, encoder_tokenizer.pkl, decoder_tokenizer.pkl, and max_lengths.pkl")

def load_model_and_metadata(model_file):
    # Load model
    model = load_model(model_file)
    
    # Load tokenizers and max lengths
    with open('encoder_tokenizer_category.pkl', 'rb') as f:
        encoder_tokenizer = pickle.load(f)
    with open('decoder_tokenizer_category.pkl', 'rb') as f:
        decoder_tokenizer = pickle.load(f)
    with open('max_lengths_category.pkl', 'rb') as f:
        lengths = pickle.load(f)
        max_input_length = lengths['max_input_length']
        max_output_length = lengths['max_output_length']
    
    print(f"Model and metadata loaded from {model_file}, encoder_tokenizer.pkl, decoder_tokenizer.pkl, and max_lengths.pkl")
    
    return model, encoder_tokenizer, decoder_tokenizer, max_input_length, max_output_length

gotModel = False

if __name__ == "__main__":
    train_flag = False  # Set to False if you want to load a pre-trained model and skip training
    
    if train_flag:
        # File path to your CSV file
        file_path = input("csv filepath: ")
        modelName = input("Model Name: ")
        
        # Load and preprocess data
        inputs, outputs = preprocess_data(file_path)
        
        # Tokenize and encode data
        max_input_length = max(len(seq.split()) for seq in inputs)
        max_output_length = max(len(seq.split()) for seq in outputs)
        
        encoder_input_data, encoder_tokenizer = tokenize_and_encode(inputs, max_input_length)
        decoder_output_data, decoder_tokenizer = tokenize_and_encode(outputs, max_output_length)
        
        # Prepare decoder input data
        decoder_input_data = np.zeros_like(decoder_output_data)
        
        # Split data into training and testing
        encoder_input_train, encoder_input_test, decoder_input_train, decoder_input_test, decoder_output_train, decoder_output_test = train_test_split(
            encoder_input_data, decoder_input_data, decoder_output_data, test_size=0.2, random_state=42
        )
        
        # Create and train the model
        input_vocab_size = len(encoder_tokenizer.word_index) + 1
        output_vocab_size = len(decoder_tokenizer.word_index) + 1
        
        model = create_seq2seq_model(input_vocab_size, output_vocab_size, max_input_length, max_output_length)
        train_model(model, encoder_input_train, decoder_input_train, decoder_output_train, epochs=10)
        
        # Save the model and metadata
        save_model_and_metadata(model, f'{modelName}.h5', encoder_tokenizer, decoder_tokenizer, max_input_length, max_output_length)
        gotModel = True
    else:
        # Load the pre-trained model and metadata
        try:
            modelName = input("Model Name: ")
            model, encoder_tokenizer, decoder_tokenizer, max_input_length, max_output_length = load_model_and_metadata(modelName)
            gotModel = True
        except:
            print("No Trained Model Found")

    # Predict a new sequence
    while True:
        if gotModel is False:
            break
        input_text = input("Line: ")
        if input_text == "q":
            break
        print(f"Input: {input_text}")
        input_seq = encoder_tokenizer.texts_to_sequences([input_text])[0]
        predicted_text = predict_sequence(model, input_seq, max_output_length, decoder_tokenizer, decoder_output_test)
        index = int(predicted_text)
        print(index)
        print(f"Predicted Output: {similar_categories[index-1][2]}")