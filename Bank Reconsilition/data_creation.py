import pandas as pd
import random
import os
from datetime import datetime, timedelta

# Create a folder named 'operations' if it does not exist
if not os.path.exists('operations'):
    os.makedirs('operations')

# List of common transaction descriptions
transaction_descriptions = [
    "Payment", "Bill", "Utility", "Fee", "Tax", "Charge", "Check", "Withdrawal", 
    "Deposit", "Transfer", "Refund", "Purchase", "Sale", "Expense", "Interest", 
    "Donation", "Service Charge", "Subscription", "Loan Repayment", "Salary", 
    "Gift", "Insurance", "Repair", "Maintenance", "Commission", "Consulting", 
    "Tuition", "Rent", "Groceries", "Entertainment", "Travel", "Healthcare"
]

# Function to create ledger transactions
def create_ledger_transactions(num_operations):
    ledger_data = []
    index = 0
    for operation_id in range(1, round(num_operations)):
        for transaction_id in range(1, random.randint(5, 31)):  # 5 to 30 transactions
            amount = random.randint(50, 500)
            transaction_type = random.choice(['Debit', 'Credit'])
            date = (datetime.now() - timedelta(days=random.randint(0, 30))).date()  # Random date in the last 30 days
            description = random.choice(transaction_descriptions)  # Random description
            ledger_data.append([transaction_id, operation_id, description, amount, transaction_type, str(date)])
            index += 1
            print(f"ledger: {index}")
    return ledger_data

# Function to create bank transactions with matching logic
def create_bank_transactions(num_operations, ledger_data):
    bank_data = []
    used_ids = set()
    index = 0
    for operation_id in range(1, round(num_operations)):
        for transaction_id in range(1, random.randint(5, 31)):  # 5 to 30 transactions
            if transaction_id % 3 == 0:  # Every third transaction will match a ledger transaction
                ledger_transaction = random.choice(ledger_data)  # Match with a ledger transaction
                amount = ledger_transaction[3]  # Match the amount
                transaction_type = ledger_transaction[4]
                description = ledger_transaction[2]  # Use the same description
                date = ledger_transaction[5]  # Use the same date
            else:
                amount = random.randint(50, 500)
                transaction_type = random.choice(['Debit', 'Credit'])
                date = (datetime.now() - timedelta(days=random.randint(0, 30))).date()  # Random date in the last 30 days
                description = random.choice(transaction_descriptions)  # Random description

            # Ensure unique transaction ID in bank data
            while transaction_id in used_ids:
                transaction_id += 1
            used_ids.add(transaction_id)

            bank_data.append([transaction_id, operation_id, description, amount, transaction_type, str(date)])
            index += 1
            print(f"Bank: {index}")
    return bank_data

# Function to create reconciliation results
def create_reconciliation_results(num_operations, ledger_data, bank_data):
    results_data = []
    index = 0
    for operation_id in range(1, round(num_operations)):
        matched_transactions = []
        not_matched_transactions = []
        
        # Create a set for easy lookup
        ledger_set = {(row[3], row[2], row[5]) for row in ledger_data if row[1] == operation_id}  # (amount, description, date)
        
        for bank_row in bank_data:
            if bank_row[1] == operation_id:
                if (bank_row[3], bank_row[2], bank_row[5]) in ledger_set:  # Match based on amount, description, and date
                    matched_transactions.append(bank_row[0])  # Store matched bank transaction ID
                else:
                    not_matched_transactions.append(bank_row[0])  # Store unmatched bank transaction ID
        index += 1
        print(f"result: {index}")
        results_data.append([operation_id, 
                             ', '.join(map(str, matched_transactions)), 
                             ', '.join(map(str, not_matched_transactions)), 
                             'Partially Matched' if len(not_matched_transactions) > 0 else 'Matched'])
        
    return results_data

# Set number of operations
num_operations = 100000  # Set this to the desired number of operations

# Create and save ledger transactions
ledger_data = create_ledger_transactions(num_operations)
ledger_df = pd.DataFrame(ledger_data, columns=['Transaction ID', 'Operation ID', 'Transaction Description', 'Amount', 'Type', 'Date'])
ledger_df.to_csv('operations/ledger_transactions.csv', index=False)

# Create and save bank transactions
bank_data = create_bank_transactions(num_operations, ledger_data)
bank_df = pd.DataFrame(bank_data, columns=['Transaction ID', 'Operation ID', 'Transaction Description', 'Amount', 'Type', 'Date'])
bank_df.to_csv('operations/bank_transactions.csv', index=False)

# Create and save reconciliation results
results_data = create_reconciliation_results(num_operations, ledger_data, bank_data)
results_df = pd.DataFrame(results_data, columns=['Operation ID', 'Matched Transactions', 'Not Matched Transactions', 'Status'])
results_df.to_csv('operations/reconciliation_results.csv', index=False)

print("CSV files have been created successfully in the 'operations' folder.")
