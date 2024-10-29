import json
import csv
import os
import random
import pandas as pd

def bst_jsonl_to_csv(output_csv_file_path, ground_truth_file_path, *jsonl_file_paths):
    all_data = []
    headers = ['TXN_DATE', 'TXN_DESC', 'category', 'debit', 'credit', 'BALANCE_AMT', 'merchant']
    
    # Read headers from the ground truth file
    if not os.path.exists(ground_truth_file_path):
        raise FileNotFoundError(f"The file {ground_truth_file_path} does not exist.")
    
    with open(ground_truth_file_path, 'r') as ground_truth_file:
        ground_truth_data = json.loads(ground_truth_file.readline())
        if "ground_truth" not in ground_truth_data:
            raise KeyError("The ground truth file does not contain the expected 'ground_truth' key.")
        
        nested_ground_truth = json.loads(ground_truth_data["ground_truth"])
        if "gt_parse" not in nested_ground_truth or "bank_stmt_entries" not in nested_ground_truth["gt_parse"] or not nested_ground_truth["gt_parse"]["bank_stmt_entries"]:
            raise KeyError("The ground truth file does not contain the expected 'gt_parse' or 'bank_stmt_entries' keys.")
    
    # Read data from each JSONL file
    for jsonl_file_path in jsonl_file_paths:
        if not os.path.exists(jsonl_file_path):
            raise FileNotFoundError(f"The file {jsonl_file_path} does not exist.")
        
        with open(jsonl_file_path, 'r') as jsonl_file:
            for line in jsonl_file:
                data = json.loads(line)
                ground_truth = json.loads(data["ground_truth"])
                if "gt_parse" not in ground_truth or "bank_stmt_entries" not in ground_truth["gt_parse"]:
                    raise KeyError(f"The JSONL file {jsonl_file_path} does not contain the expected 'gt_parse' or 'bank_stmt_entries' keys.")
                for entry in ground_truth["gt_parse"]["bank_stmt_entries"]:
                    # Add default values for 'merchant' and 'category' if they do not exist
                    entry['merchant'] = entry.get('merchant', 'Unknown')
                    entry['category'] = entry.get('category', 'Uncategorized')
                    
                    # Map 'DEPOSIT_AMT' to 'debit' and 'WITHDRAWAL_AMT' to 'credit'
                    entry['debit'] = entry.pop('DEPOSIT_AMT', '')
                    entry['credit'] = entry.pop('WITHDRAWAL_AMT', '')
                    
                    all_data.append(entry)
    
    # Write combined data to a single CSV file
    with open(output_csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the header
        csv_writer.writerow(headers)
        
        # Write the data
        for data in all_data:
            csv_writer.writerow([data.get(header, "") for header in headers])


# Create 'debit' and 'credit' columns based on 'category'
def categorize_transaction(row):
    category = row['category'].lower()  # Normalize the category to lowercase
    amount = row['amount']  # Get the transaction amount
    
    # Initialize debit and credit amounts
    debit_amount = ''
    credit_amount = ''

    # Classify based on category
    if 'deposit' in category or 'payroll' in category or 'internal account transfer' in category:
        debit_amount = amount  # Debit for increases in the account (deposits)
    elif 'check withdrawal' in category or 'transfer debit' in category or 'loan' in category or 'shops' in category or 'atm' in category:
        credit_amount = amount  # Credit for outflows (decreases)
    elif 'transfer credit' in category:
        debit_amount = amount  # Debit for positive transfers
    elif 'third party' in category:
        description = row['TXN_DESC'].lower()  # Normalize the transaction description to lowercase
        
        # Check for keywords in the transaction description
        if any(keyword in description for keyword in ['payment from', 'cash in', 'deposit from']):
            debit_amount = amount  # Debit for positive transactions
        elif any(keyword in description for keyword in ['*cash out', 'cash out', 'credit', 'withdrawal']):
            credit_amount = amount  # Credit for negative transactions
        else:
            # Default to treating as credit for unknown descriptions
            credit_amount = amount  
    else:
        # Default to treating as credit for unknown categories
        credit_amount = amount  
    
    return debit_amount, credit_amount  # Return both debit and credit amounts


def determine_transaction_type(row):
        category = row['category'].lower()
        description = row['TXN_DESC'].lower()
        debit = row['debit']
        credit = row['credit']
        
        if any(keyword in category for keyword in ['loan', 'equity', 'royalty']) or any(keyword in description for keyword in ['loan', 'equity', 'royalty']):
            return 'Unrecognized'
        elif debit not in ['', 'N/A', 0]:
            return 'Income'
        elif credit not in ['', 'N/A', 0]:
            return 'Expense'
        else:
            return 'Unrecognized'

# Function to convert Parquet to CSV with specific headers and conditions
def TD_parquet_to_csv(input_parquet_file_path, output_csv_file_path):
    df = pd.read_parquet(input_parquet_file_path)
    
    # Select only the required columns and rename them
    df = df[['txn_date', 'description', 'amount', 'category']]
    df = df.rename(columns={'txn_date': 'TXN_DATE', 'description': 'TXN_DESC'})

    df['debit'], df['credit'] = zip(*df.apply(categorize_transaction, axis=1))
    
    # Add 'BALANCE_AMT' with random amounts
    df['BALANCE_AMT'] = pd.Series([round(random.uniform(1000, 5000), 2) for _ in range(len(df))])
    
    # Add 'merchant' column with default value 'N/A'
    df['merchant'] = 'N/A'
    
    df['transaction_type'] = df.apply(determine_transaction_type, axis=1)
    
    # Drop the 'amount' column
    df = df.drop(columns=['amount'])
    
    # Save to CSV
    df.to_csv(output_csv_file_path, index=False)

# Convert the specified Parquet file to CSV
TD_parquet_to_csv(
    "/media/alphi/Alphi/Python/Python Project/Spensibily_OCR/Product_Name_And_Price_Classification/Dataset/Bank/Transaction-Data/all/train-00000-of-00001.parquet",
    "/media/alphi/Alphi/Python/Python Project/Spensibily_OCR/Product_Name_And_Price_Classification/Dataset/Bank/Transaction-Data/all/train-00000-of-00001.csv"
)


# Example usage
bst_jsonl_to_csv(
    "/media/alphi/Alphi/Python/Python Project/Spensibily_OCR/Product_Name_And_Price_Classification/Dataset/Bank/bank_statements_transactions/combined_metadata.csv",
    "/media/alphi/Alphi/Python/Python Project/Spensibily_OCR/Product_Name_And_Price_Classification/Dataset/Bank/bank_statements_transactions/test/metadata.jsonl",
    "/media/alphi/Alphi/Python/Python Project/Spensibily_OCR/Product_Name_And_Price_Classification/Dataset/Bank/bank_statements_transactions/train/metadata.jsonl",
    "/media/alphi/Alphi/Python/Python Project/Spensibily_OCR/Product_Name_And_Price_Classification/Dataset/Bank/bank_statements_transactions/validation/metadata.jsonl"
)