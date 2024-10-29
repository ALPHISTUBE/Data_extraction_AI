import pandas as pd
import numpy as np
from random import choice, randint
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# Function to simulate multiple operations with sequences for Seq2Seq
def create_sample_operations_with_sequences(num_operations):
    operations = []
    ledger_types = ['Debit', 'Credit']
    
    for operation_id in range(1, num_operations + 1):
        # Generate random number of transactions
        num_transactions = randint(2, 5)
        transaction_sequence = []
        status_sequence = []

        # Create sequences for transactions
        for transaction_id in range(1, num_transactions + 1):
            ledger_type = choice(ledger_types)
            ledger_description = f'Ledger Transaction {transaction_id} for Operation {operation_id}'
            bank_description = f'Bank Transaction {transaction_id} for Operation {operation_id}'
            transaction_sequence.append(f'{ledger_description}, {bank_description}')
            
            # Simulate status
            status = 'Matched' if transaction_id % 2 == 1 else 'Not Matched'
            status_sequence.append(status)

        operations.append({
            'Operation ID': operation_id,
            'Transaction Sequence': ' | '.join(transaction_sequence),
            'Status Sequence': ', '.join(status_sequence)
        })

    return pd.DataFrame(operations)

# Generate operations
num_operations = 100000  # Set this to your desired number of operations
df_operations = create_sample_operations_with_sequences(num_operations)
df_operations.to_csv('operations_with_sequences.csv', index=False)

# Load the combined operations data
df = pd.read_csv('operations_with_sequences.csv')

# Split the sequences into input (transaction) and output (status)
X = df['Transaction Sequence'].values
y = df['Status Sequence'].values

# Tokenize the input sequences
tokenizer_input = Tokenizer()
tokenizer_input.fit_on_texts(X)
input_sequences = tokenizer_input.texts_to_sequences(X)

# Tokenize the output sequences
tokenizer_output = Tokenizer()
tokenizer_output.fit_on_texts(y)
output_sequences = tokenizer_output.texts_to_sequences(y)

# Pad sequences to the same length
max_input_length = max(len(seq) for seq in input_sequences)
max_output_length = max(len(seq) for seq in output_sequences)

X_pad = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
y_pad = pad_sequences(output_sequences, maxlen=max_output_length, padding='post')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_pad, test_size=0.2, random_state=42)

# Define parameters
latent_dim = 256  # Dimensionality of the latent space
num_encoder_tokens = len(tokenizer_input.word_index) + 1
num_decoder_tokens = len(tokenizer_output.word_index) + 1

# Define the encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Define the decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit([X_train, y_train[:, :-1]], y_train[:, 1:, np.newaxis],
          batch_size=64,
          epochs=100,
          validation_data=([X_test, y_test[:, :-1]], y_test[:, 1:, np.newaxis]))

# Save the trained model
model.save('seq2seq_reconciliation_model.h5')
