import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import pickle
import os

# Set a seed for reproducibility
np.random.seed(42)

# Load and preprocess the dataset
def load_data(file_path):
    # Read the CSV file with header
    data = pd.read_csv(file_path)

    # Ensure the expected columns exist
    if not {'Description', 'TransactionType'}.issubset(data.columns):
        raise ValueError("CSV must contain 'Description' and 'TransactionType' columns.")

    # Map the TransactionType to numerical labels
    data['Label'] = data['TransactionType'].map({'Income': 0, 'Expense': 1, 'Unrecognized': 2})
    
    # Drop any rows with NaN in the 'Label' column
    data.dropna(subset=['Label'], inplace=True)

    # Reset index after dropping rows
    data.reset_index(drop=True, inplace=True)

    return data

# Reinforcement Learning Agent
class RLAgent:
    def __init__(self):
        self.vectorizer = CountVectorizer()  # Initialize CountVectorizer
        self.model = GaussianNB()
        self.experience = []
        self.reinforcement_count = 0

    def train(self, data):
        # Vectorize descriptions
        X = self.vectorizer.fit_transform(data['Description']).toarray()  # Convert to dense array
        y = data['Label'].values  # Get the labels

        self.model.fit(X, y)

    def predict(self, description):
        # Transform input description into the same vector space
        description_vector = self.vectorizer.transform([description]).toarray()  # Convert to dense array
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

# Main function
def main(file_path, model_filename):
    # Check if model exists
    if os.path.exists(model_filename):
        agent = load_model(model_filename)
        print("Loaded existing model.")
    else:
        data = load_data(file_path)
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
            agent.store_experience(user_input, actual_label)  # Store the experience
            agent.learn()  # Reinforce learning
        else:
            print("Invalid transaction type. Please enter again.")

    # Save model
    save_model(agent, model_filename)
    print("Model saved successfully!")

# Example usage
if __name__ == "__main__":
    dataset_path = "income_expanse_dataset.csv"  # Specify your dataset path
    model_filename = "rl_transaction_classifier.pkl"
    main(dataset_path, model_filename)
