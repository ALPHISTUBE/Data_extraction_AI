import pandas as pd
import numpy as np
from random import choice, randint

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
df_operations = create_sample_operations_with_sequences(2)  # Change to 100000 for real use
print(df_operations)

# Save the operations to a CSV file
df_operations.to_csv('operations_with_sequences.csv', index=False)
