import pandas as pd

# Load the CSV files into DataFrames
bookPassPath = "Book_Pass.csv"
bankPassPath = "Bank_Pass.csv"

receipt_df = pd.read_csv(bookPassPath)
bank_df = pd.read_csv(bankPassPath)

# Ensure the date format is consistent and Amount is numeric
receipt_df['Date'] = pd.to_datetime(receipt_df['Date'], dayfirst=True, errors='coerce')
bank_df['Date'] = pd.to_datetime(bank_df['Date'], dayfirst=True, errors='coerce')

receipt_df['Amount'] = pd.to_numeric(receipt_df['Amount'], errors='coerce')
bank_df['Amount'] = pd.to_numeric(bank_df['Amount'], errors='coerce')

# Step 1: Update book pass amounts to match bank pass amounts
merged_df = pd.merge(receipt_df, bank_df, on=['Date', 'Transaction Type'], suffixes=('_book', '_bank'), how='left', indicator=True)

for idx, row in merged_df.iterrows():
    if row['_merge'] == 'both':
        if row['Amount_book'] != row['Amount_bank']:
            receipt_df.loc[receipt_df.index == idx, 'Amount'] = row['Amount_bank']

# Step 2: Remove unmatched Check Withdrawals from book pass
book_check_withdrawal = receipt_df[receipt_df['Transaction Type'] == 'Check Withdrawal']
bank_check_withdrawal = bank_df[bank_df['Transaction Type'] == 'Check Withdrawal']

matching_check_withdrawal = pd.merge(book_check_withdrawal, bank_check_withdrawal, on=['Date', 'Transaction Type', 'Amount'], how='left', indicator=True)
non_matching_check_withdrawal = book_check_withdrawal[~book_check_withdrawal.index.isin(matching_check_withdrawal.index)]
removed_check_withdrawal = non_matching_check_withdrawal.copy()
removed_check_withdrawal['Status'] = 'Removed'

# Remove unmatched Check Withdrawal from book pass
updated_receipt_df = receipt_df[~receipt_df.index.isin(non_matching_check_withdrawal.index)]

# Step 3: Create unmatched tables
reconciliation = pd.merge(bank_df, updated_receipt_df, on=['Date', 'Amount', 'Transaction Type'], how='outer', indicator=True)
unmatched_book_pass = reconciliation[reconciliation['_merge'] == 'left_only']
print(reconciliation)
unmatched_bank_pass = reconciliation[reconciliation['_merge'] == 'right_only']

# Step 4: Add unmatched amounts from book to bank and from bank to book
unmatched_book_pass['Status'] = 'From Book'
unmatched_bank_pass['Status'] = 'From Bank'

# Adjust amounts for reconciliation
def adjust_amount(df):
    df['Amount'] = df.apply(lambda row: row['Amount'] if 'Deposit' in row['Transaction Type'] else -row['Amount'], axis=1)
    return df

unmatched_book_pass = unmatched_book_pass[['Date', 'Transaction Type', 'Amount', 'Status']]
unmatched_bank_pass = unmatched_bank_pass[['Date', 'Transaction Type', 'Amount', 'Status']]

# Merge unmatched transactions into the main tables
updated_receipt_df = pd.concat([updated_receipt_df, unmatched_book_pass]).reset_index(drop=True)
updated_bank_df = pd.concat([bank_df, unmatched_bank_pass]).reset_index(drop=True)

# Adjust amounts in the final tables
updated_receipt_df = adjust_amount(updated_receipt_df)
updated_bank_df = adjust_amount(updated_bank_df)



# Step 5: Calculate sums and check reconciliation
sum_updated_receipt = round(updated_receipt_df['Amount'].sum(), 2)
sum_updated_bank = round(updated_bank_df['Amount'].sum(), 2)

# Display results
print("\nBank Transactions:")
print(bank_df)

print("\nReceipt Transactions:")
print(receipt_df)

print("\nRemoved Check Withdrawal Transactions:")
print(removed_check_withdrawal[['Date', 'Transaction Type', 'Amount', 'Status']])

print("\nMarge Transactions:")
print(reconciliation)

print("\nUnmatched Book Pass Transactions Added to Bank Table:")
print(unmatched_book_pass)

print("\nUnmatched Bank Pass Transactions Added to Book Table:")
print(unmatched_bank_pass)

print("\nUpdated Bank Transactions:")
print(updated_bank_df)

print("\nUpdated Receipt Transactions:")
print(updated_receipt_df)

print(f"\nSum of Updated Receipt Transactions: {sum_updated_receipt:.2f}")
print(f"\nSum of Updated Bank Transactions: {sum_updated_bank:.2f}")

if sum_updated_receipt == sum_updated_bank:
    print("\nBank reconciliation succeeded.")
else:
    print("\nBank reconciliation failed.")

# Save the final tables to CSV files if needed
updated_receipt_df.to_csv('updated_receipt.csv', index=False)
updated_bank_df.to_csv('updated_bank.csv', index=False)
