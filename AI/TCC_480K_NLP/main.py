import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pickle
import threading

payment_categories = [
    [0, "Payment"],
    [1, "Deposit"],
    [2, "Loan Repayment"],
    [3, "Gift"],
    [4, "Rent"],
    [5, "Refund"],
    [6, "Services"],
    [7, "Withdrawal"],
    [8, "Donation"],
    [9, "Purchase"],
    [10, "Unrecognized"],
]

similar_category = [
    [0, "", "Rent", "Rent"],
    [1, "Utilities", "Utilities", "Services"],
    [2, "Utilities", "Utilities (Electricity)", "Services"],
    [3, "Utilities", "Utilities (Water)", "Services"],
    [4, "Utilities", "Utilities (Gas)", "Services"],
    [5, "Telecommunication Services", "Internet Service", "Services"],
    [6, "Telecommunication Services", "Telephone Service", "Services"],
    [7, "", "POS System Fees", "Payment"],
    [8, "Supermarkets and Groceries", "Inventory Purchases", "Purchase"],
    [9, "Shops", "Inventory Purchases", "Purchase"],
    [10, "", "Shipping & Freight Costs", "Services"],
    [11, "", "Packaging Supplies", "Services"],
    [12, "", "Store Supplies", "Purchase"],
    [13, "", "Cleaning Supplies", "Purchase"],
    [14, "", "Marketing & Advertising", "Services"],
    [15, "", "Website Maintenance", "Services"],
    [16, "", "E-commerce Platform Fees", "Payment"],
    [17, "Bank Fees", "Merchant Processing Fees", "Payment"],
    [18, "Payroll", "Payroll", "Deposit"],
    [19, "", "Employee Benefits", "Services"],
    [20, "", "Employee Training", "Services"],
    [21, "Clothing and Accessories", "Uniforms", "Purchase"],
    [22, "", "Security Services", "Services"],
    [23, "Insurance", "Insurance", "Payment"],
    [24, "Insurance", "Insurance (Property)", "Payment"],
    [25, "Insurance", "Insurance (Liability)", "Payment"],
    [26, "Insurance", "Insurance (Workers' Compensation)", "Payment"],
    [27, "Insurance", "Insurance (Health)", "Payment"],
    [28, "Tax Refund", "Tax Refund", "Refund"],
    [29, "Tax Refund", "Taxes (Sales)", "Refund"],
    [30, "Tax Refund", "Taxes (Property)", "Refund"],
    [31, "Tax Refund", "Taxes (Payroll)", "Refund"],
    [32, "", "Accounting Services", "Services"],
    [33, "", "Legal Services", "Services"],
    [34, "", "Professional Fees", "Payment"],
    [35, "Bank Fees", "Banking Fees", "Payment"],
    [36, "Interest", "Interest Expenses", "Payment"],
    [37, "Loans", "Loan Repayments", "Loan Repayment"],
    [38, "Loan payment", "Loan Repayments", "Loan Repayment"],
    [39, "", "Equipment Leasing", "Rent"],
    [40, "", "Equipment Repairs", "Payment"],
    [41, "", "Store Decorations", "Purchase"],
    [42, "", "Office Supplies", "Purchase"],
    [43, "Travel", "Travel Expenses", "Purchase"],
    [44, "", "Vehicle Leasing", "Rent"],
    [45, "", "Vehicle Maintenance", "Payment"],
    [46, "Gas Stations", "Fuel Costs", "Purchase"],
    [47, "", "Delivery Expenses", "Purchase"],
    [48, "", "Customer Returns", "Refund"],
    [49, "", "Discounts Given", "Refund"],
    [50, "", "Employee Discounts", "Payment"],
    [51, "", "Refunds Issued", "Refund"],
    [52, "", "Loyalty Program Costs", "Payment"],
    [53, "", "Gift Wrapping Supplies", "Gift"],
    [54, "", "Gift Card Processing Fees", "Gift"],
    [55, "", "Bad Debts", "Payment"],
    [56, "", "Depreciation", "Payment"],
    [57, "", "Amortization", "Payment"],
    [58, "Charity donation", "Charitable Donations", "Donation"],
    [59, "", "Community Sponsorships", "Donation"],
    [60, "Restaurants", "Staff Meals", "Purchase"],
    [61, "Food and Beverage Services", "Staff Meals", "Purchase"],
    [62, "", "Holiday Decorations", "Purchase"],
    [63, "", "Seasonal Displays", "Purchase"],
    [64, "", "Window Display Costs", "Purchase"],
    [65, "", "Signage Costs", "Purchase"],
    [66, "", "Music Licensing Fees", "Payment"],
    [67, "", "Pest Control", "Services"],
    [68, "", "Landscaping", "Services"],
    [69, "", "Snow Removal", "Services"],
    [70, "", "Waste Disposal", "Services"],
    [71, "", "Recycling Services", "Services"],
    [72, "", "Warehouse Rent", "Rent"],
    [73, "", "Warehouse Supplies", "Purchase"],
    [74, "", "Warehouse Utilities", "Payment"],
    [75, "", "Cash Register Supplies", "Purchase"],
    [76, "ATM", "ATM Fees", "Payment"],
    [77, "", "Credit Card Fees", "Payment"],
    [78, "", "Debit Card Fees", "Payment"],
    [79, "", "Employee Recruitment", "Payment"],
    [80, "", "Employee Onboarding", "Payment"],
    [81, "", "Training Materials", "Purchase"],
    [82, "", "Conferences & Seminars", "Payment"],
    [83, "", "Subscriptions & Dues", "Purchase"],
    [84, "", "Membership Fees", "Payment"],
    [85, "", "IT Support Services", "Services"],
    [86, "", "Software Licenses", "Payment"],
    [87, "", "Cloud Storage Services", "Payment"],
    [88, "", "Data Backup Services", "Payment"],
    [89, "", "Printing & Copying", "Payment"],
    [90, "", "Office Equipment", "Purchase"],
    [91, "", "Mail & Courier Services", "Services"],
    [92, "", "Customer Service Expenses", "Services"],
    [93, "", "Inventory Shrinkage", "Payment"],
    [94, "", "Product Sampling Costs", "Purchase"],
    [95, "", "Store Fixtures", "Purchase"],
    [96, "", "In-store Events", "Purchase"],
    [97, "", "Promotional Materials", "Purchase"],
    [98, "", "Survey & Feedback Tools", "Purchase"],
    [99, "", "Business Cards", "Purchase"],
    [100, "", "Loyalty Cards", "Purchase"],
    [101, "", "Business Licenses & Permits", "Payment"],
    [102, "", "Environmental Fees", "Payment"],
    [103, "", "Employee Wellness Programs", "Payment"],
    [104, "Transfer Credit", "Transfer Credit", "Withdrawal"],
    [105, "Transfer Deposit", "Transfer Deposit", "Deposit"],
    [106, "Check Deposit", "Check Deposit", "Deposit"],
    [107, "Third Party", "Third Party", "Withdrawal"],
    [108, "", "Internal Account Transfer", "Deposit"],
    [109, "Digital Entertainment", "Arts and Entertainment", "Purchase"],
    [110, "Gyms and Fitness Centers", "Gyms and Fitness Centers", "Purchase"],
    [111, "Department Stores", "Inventory Purchases", "Purchase"],
    [112, "Healthcare", "Healthcare", "Purchase"],
    [113, "Service", "Service", "Services"],
    [114, "Arts and Entertainment", "Arts and Entertainment", "Purchase"],
    [115, "Convenience Stores", "Convenience Stores", "Purchase"],
    [116, "Payment", "Payment", "Payment"],
    [117, "Personal transfer", "Personal transfer", "Unrecognized"],
    [118, "Gift", "Gift", "Gift"],
    [119, "Stock purchase", "Stock purchase", "Purchase"],
    [120, "Business investment", "Business investment", "Purchase"],
    [121, "Real estate purchase", "Real estate purchase", "Purchase"],
    [122, "Loan payment", "Loan payment", "Loan Repayment"],
    [123, "Transfer Debit", "Transfer Debit", "Deposit"],
    [124, "Transfer Credit", "Transfer Credit", "Withdrawal"],
    [117, "Internal Account Transfer", "Personal transfer", "Deposit"],
]

transaction_types = [
    [0, "Income"],
    [1, "Expense"],
    [2, "Unrecognized"]
]

# Data Preprocessing
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    inputs = df['TXN_DESC'].tolist()
    categories = df['category'].astype(int).tolist()
    transaction_types = df['transaction_type'].astype(int).tolist()
    purposes = df['purpose'].astype(int).tolist()
    
    # Merge categories, transaction_types, and purposes as a single string like "12 1 2"
    outputs = [f"{category} {transaction_type} {purpose}" for category, transaction_type, purpose in zip(categories, transaction_types, purposes)]
    
    return inputs, outputs

# Tokenization and padding
def tokenize_and_encode(texts, max_length, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    return padded_sequences, tokenizer

# Model Creation
def create_seq2seq_model(input_vocab_size, output_vocab_size, input_seq_length, output_seq_length, embedding_dim=256, lstm_units=512):
    # Encoder
    encoder_inputs = Input(shape=(input_seq_length,))
    encoder_embedding = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm, forward_h, forward_c = LSTM(lstm_units, return_state=True, return_sequences=True)(encoder_embedding)

    # Decoder
    decoder_inputs = Input(shape=(output_seq_length,))
    decoder_embedding = Embedding(output_vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm, decoder_h, decoder_c = LSTM(lstm_units, return_sequences=True, return_state=True)(decoder_embedding, initial_state=[forward_h, forward_c])
    
    # Attention
    attention = Attention()([decoder_lstm, encoder_lstm])
    concat = Concatenate(axis=-1)([decoder_lstm, attention])
    
    # Dense Output Layer
    output = Dense(output_vocab_size, activation='softmax')(concat)
    
    # Create model
    model = Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Model Training
def train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, epochs=10, batch_size=64):
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2
    )

# Prediction
def predict_sequence(model, input_seq, max_output_length, tokenizer):
    # Pad input sequence
    input_seq = pad_sequences([input_seq], maxlen=max_input_length, padding='post')
    
    # Initialize decoder input with zeros
    decoder_input = np.zeros((1, max_output_length))
    
    # Predict the output sequence
    prediction = model.predict([input_seq, decoder_input])
    
    # Extract indices of the most probable tokens
    predicted_indices = np.argmax(prediction, axis=-1)[0]
    
    # Convert indices back to text
    predicted_text = tokenizer.sequences_to_texts([predicted_indices])[0]
    
    return predicted_text

# Save model and tokenizers with TCC_480K_ prefix
def save_model_and_metadata(model, model_file, encoder_tokenizer, decoder_tokenizer, max_input_length, max_output_length):
    model.save(model_file)
    with open('TCC_480K_encoder_tokenizer.pkl', 'wb') as f:
        pickle.dump(encoder_tokenizer, f)
    with open('TCC_480K_decoder_tokenizer.pkl', 'wb') as f:
        pickle.dump(decoder_tokenizer, f)
    with open('TCC_480K_max_lengths.pkl', 'wb') as f:
        pickle.dump({'max_input_length': max_input_length, 'max_output_length': max_output_length}, f)

# Load model and tokenizers with TCC_480K_ prefix
def load_model_and_metadata(model_file):
    model = load_model(model_file)
    with open('TCC_480K_encoder_tokenizer.pkl', 'rb') as f:
        encoder_tokenizer = pickle.load(f)
    with open('TCC_480K_decoder_tokenizer.pkl', 'rb') as f:
        decoder_tokenizer = pickle.load(f)
    with open('TCC_480K_max_lengths.pkl', 'rb') as f:
        lengths = pickle.load(f)
    return model, encoder_tokenizer, decoder_tokenizer, lengths['max_input_length'], lengths['max_output_length']

# Main Workflow
if __name__ == "__main__":
    train_flag = True  # Change this to True to train a new model
    
    if train_flag:
        # File path and model name
        file_path = '/media/alphi/Alphi/Python/Python Project/Spensibily_OCR/AI/TCC_480K_NLP/output_file_480k.csv'
        modelName = "TCC_480K_NLP"
        
        # Load and preprocess data
        inputs, outputs = preprocess_data(file_path)
        
        # Determine max input/output lengths
        max_input_length = max(len(seq.split()) for seq in inputs)
        max_output_length = max(len(seq.split()) for seq in outputs)
        
        # Tokenize and encode the data
        encoder_input_data, encoder_tokenizer = tokenize_and_encode(inputs, max_input_length)
        decoder_output_data, decoder_tokenizer = tokenize_and_encode(outputs, max_output_length)
        
        # Prepare decoder input data (shifted version of decoder_output_data)
        decoder_input_data = np.zeros_like(decoder_output_data)
        
        # Split into training and testing sets
        encoder_input_train, encoder_input_test, decoder_input_train, decoder_input_test, decoder_output_train, decoder_output_test = train_test_split(
            encoder_input_data, decoder_input_data, decoder_output_data, test_size=0.2, random_state=42
        )
        
        # Create the model
        input_vocab_size = len(encoder_tokenizer.word_index) + 1
        output_vocab_size = len(decoder_tokenizer.word_index) + 1
        model = create_seq2seq_model(input_vocab_size, output_vocab_size, max_input_length, max_output_length)
        
        # Train the model in a separate thread
        train_thread = threading.Thread(target=train_model, args=(model, encoder_input_train, decoder_input_train, decoder_output_train, 10))
        train_thread.start()
        train_thread.join()
        
        # Save the model and metadata
        save_model_and_metadata(model, f'{modelName}.h5', encoder_tokenizer, decoder_tokenizer, max_input_length, max_output_length)
    
    else:
        # Load the pre-trained model
        try:
            modelName = "TCC_480K_NLP"
            model, encoder_tokenizer, decoder_tokenizer, max_input_length, max_output_length = load_model_and_metadata(f'{modelName}.h5')
        except:
            print("No trained model found.")
            exit()

    # Predict a sequence
    while True:
        input_text = input("Input Sequence: ")
        if input_text.lower() == "q":
            break
        input_seq = encoder_tokenizer.texts_to_sequences([input_text])[0]
        predicted_text = predict_sequence(model, input_seq, max_output_length, decoder_tokenizer)
        print(f"Predicted Output: {predicted_text}")
