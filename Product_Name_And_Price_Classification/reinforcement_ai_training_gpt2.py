import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# Set a seed for reproducibility
np.random.seed(42)

# Load and preprocess the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)

    if not {'Description', 'TransactionType'}.issubset(data.columns):
        raise ValueError("CSV must contain 'Description' and 'TransactionType' columns.")

    data['Label'] = data['TransactionType'].map({'Income': 0, 'Expense': 1, 'Unrecognized': 2})
    data.dropna(subset=['Label'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data

# Reinforcement Learning Agent
class RLAgent:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = GaussianNB()
        self.experience = []
        self.reinforcement_count = 0

    def train(self, data):
        X = self.vectorizer.fit_transform(data['Description']).toarray()
        y = data['Label'].values
        self.model.fit(X, y)

    def predict(self, description):
        description_vector = self.vectorizer.transform([description]).toarray()
        return self.model.predict(description_vector)[0]

    def reward(self, prediction, actual):
        return 1 if prediction == actual else -1

    def learn(self):
        for experience in self.experience:
            description, actual = experience
            prediction = self.predict(description)
            rwd = self.reward(prediction, actual)
            if rwd > 0:
                self.reinforcement_count += 1
                self.train(pd.DataFrame(self.experience, columns=['Description', 'Label']))

    def store_experience(self, description, actual):
        self.experience.append((description, actual))

# Function to save the trained model
def save_model(agent, filename):
    with open(filename, 'wb') as file:
        pickle.dump(agent, file)

# Function to load the trained model
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Load the model and tokenizer for CodeLlama
tokenizer = LlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")

# Function to generate transaction descriptions
def generate_transaction_descriptions(category, num_samples=5):
    descriptions = []
    for _ in range(num_samples):
        input_text = f"Generate a {category.lower()} transaction description:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate output using CodeLlama
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        
        # Clean the generated text to remove unwanted patterns
        descriptions.append(generated_text.replace(input_text, '').strip())
    return descriptions

# Main function
def main(dataset_path, model_filename):
    # Generate synthetic data if the dataset doesn't exist
    if not os.path.exists(dataset_path):
        N = 10  # Number of samples to generate
        income_samples = generate_transaction_descriptions("Income", N)
        expense_samples = generate_transaction_descriptions("Expense", N)
        unrecognized_samples = generate_transaction_descriptions("Royalty, Equity, Liability, Loan", N)

        data = {
            "Description": income_samples + expense_samples + unrecognized_samples,
            "TransactionType": ["Income"] * len(income_samples) + ["Expense"] * len(expense_samples) + ["Unrecognized"] * len(unrecognized_samples)
        }
        df = pd.DataFrame(data)
        df.to_csv(dataset_path, index=False)

    # Load and train the RL agent
    if os.path.exists(model_filename):
        agent = load_model(model_filename)
        print("Loaded existing model.")
    else:
        data = load_data(dataset_path)
        agent = RLAgent()
        agent.train(data)  # Initial training
        print("Trained new model.")

    # User testing loop
    while True:
        user_input = input("Enter transaction description (or 'q' to quit): ")
        
        if user_input.lower() == 'q':
            print(f"Reinforcement occurred {agent.reinforcement_count} times during the session.")
            break
        
        prediction = agent.predict(user_input)
        print(f"Predicted Transaction Type: {['Income', 'Expense', 'Unrecognized'][prediction]}")

        actual_input = input("Enter actual transaction type (Income, Expense, Unrecognized): ")
        if actual_input in ['Income', 'Expense', 'Unrecognized']:
            actual_label = {'Income': 0, 'Expense': 1, 'Unrecognized': 2}[actual_input]
            agent.store_experience(user_input, actual_label)
            agent.learn()
        else:
            print("Invalid transaction type. Please enter again.")

    # Save model
    save_model(agent, model_filename)
    print("Model saved successfully!")

# Example usage
if __name__ == "__main__":
    dataset_path = "synthetic_transactions.csv"  # Specify your dataset path
    model_filename = "rl_transaction_classifier_V2.pkl"
    main(dataset_path, model_filename)
