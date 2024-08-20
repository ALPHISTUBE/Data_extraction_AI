import pandas as pd
from django.shortcuts import render
from django.core.files.storage import default_storage
from .forms import UploadFileForm
from django.core.files.storage import FileSystemStorage
import os

def reconcile_files(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            book_file = request.FILES['book_file']
            bank_file = request.FILES['bank_file']
            
            # Save uploaded files            
            # Save uploaded files
            book_file_path = default_storage.save('' + book_file.name, book_file)
            bank_file_path = default_storage.save('' + bank_file.name, bank_file)
            
            book_file_path = 'media/' + book_file_path
            bank_file_path = 'media/' + bank_file_path

            # Process files
            result = process_files(book_file_path, bank_file_path)
            
            # Delete files after processing
            default_storage.delete(book_file_path)
            default_storage.delete(bank_file_path)
            
            return render(request, 'results.html', result)
    else:
        form = UploadFileForm()
    
    return render(request, 'upload.html', {'form': form})

def process_files(book_file_path, bank_file_path):
    # Load the CSV files into DataFrames
    receipt_df = pd.read_csv(book_file_path)
    bank_df = pd.read_csv(bank_file_path)

    # Ensure the date format is consistent and Amount is numeric
    # receipt_df['Date'] = pd.to_datetime(receipt_df['Date'], dayfirst=True, errors='coerce')
    # bank_df['Date'] = pd.to_datetime(bank_df['Date'], dayfirst=True, errors='coerce')

    receipt_df['Amount'] = pd.to_numeric(receipt_df['Amount'], errors='coerce')
    bank_df['Amount'] = pd.to_numeric(bank_df['Amount'], errors='coerce')

    receipt_df.rename(columns={'Transaction Type': 'Transaction_Type'}, inplace=True)
    bank_df.rename(columns={'Transaction Type': 'Transaction_Type'}, inplace=True)

    # Step 1: Update book pass amounts to match bank pass amounts
    merged_df = pd.merge(receipt_df, bank_df, on=['Date', 'Transaction_Type'], suffixes=('_book', '_bank'), how='left', indicator=True)
    for idx, row in merged_df.iterrows():
        if row['_merge'] == 'both':
            if row['Amount_book'] != row['Amount_bank']:
                receipt_df.loc[receipt_df.index == idx, 'Amount'] = row['Amount_bank']
                receipt_df.loc[receipt_df.index == idx, 'Status'] = 'Amount Updated'

    # Step 2: Remove unmatched Check Withdrawals from book pass
    book_check_withdrawal = receipt_df[receipt_df['Transaction_Type'] == 'Check Withdrawal']
    bank_check_withdrawal = bank_df[bank_df['Transaction_Type'] == 'Check Withdrawal']
    matching_check_withdrawal = pd.merge(book_check_withdrawal, bank_check_withdrawal, on=['Date', 'Transaction_Type', 'Amount'], how='left', indicator=True)
    non_matching_check_withdrawal = book_check_withdrawal[~book_check_withdrawal.index.isin(matching_check_withdrawal.index)]
    removed_check_withdrawal = non_matching_check_withdrawal.copy()
    removed_check_withdrawal['Status'] = 'Removed'

    # Remove unmatched Check Withdrawal from book pass
    updated_receipt_df = receipt_df[~receipt_df.index.isin(non_matching_check_withdrawal.index)]

    # Step 3: Create unmatched tables
    reconciliation = pd.merge(bank_df, updated_receipt_df, on=['Date', 'Amount', 'Transaction_Type'], how='outer', indicator=True)
    unmatched_book_pass = reconciliation[reconciliation['_merge'] == 'left_only']
    unmatched_bank_pass = reconciliation[reconciliation['_merge'] == 'right_only']

    # Step 4: Add unmatched amounts from book to bank and from bank to book
    unmatched_book_pass['Status'] = 'From Bank'
    unmatched_bank_pass['Status'] = 'From Book'

    # Adjust amounts for reconciliation
    def adjust_amount(df):
        df['Amount'] = df.apply(lambda row: row['Amount'] if 'Deposit' in row['Transaction_Type'] else -row['Amount'], axis=1)
        return df

    unmatched_book_pass = unmatched_book_pass[['Date', 'Transaction_Type', 'Amount', 'Status']]
    unmatched_bank_pass = unmatched_bank_pass[['Date', 'Transaction_Type', 'Amount', 'Status']]

    # Merge unmatched transactions into the main tables
    updated_receipt_df = pd.concat([updated_receipt_df, unmatched_book_pass]).reset_index(drop=True)
    updated_bank_df = pd.concat([bank_df, unmatched_bank_pass]).reset_index(drop=True)

    # Adjust amounts in the final tables
    updated_receipt_df = adjust_amount(updated_receipt_df)
    updated_bank_df = adjust_amount(updated_bank_df)

    # Step 5: Assign 'Matched' status to all matched transactions in updated tables
    updated_receipt_df['Status'] = updated_receipt_df['Status'].fillna('Matched')
    updated_bank_df['Status'] = updated_bank_df['Status'].fillna('Matched')

    # Step 6: Calculate sums and check reconciliation
    sum_updated_receipt = round(updated_receipt_df['Amount'].sum(), 2)
    sum_updated_bank = round(updated_bank_df['Amount'].sum(), 2)

    # Return results for rendering
    return {
        'book_pass': receipt_df,
        'bank_pass': bank_df,
        'removed_check_withdrawal': removed_check_withdrawal[['Date', 'Transaction_Type', 'Amount', 'Status']],
        'unmatched_book_pass': unmatched_book_pass,
        'unmatched_bank_pass': unmatched_bank_pass,
        'updated_receipt_df': updated_receipt_df,
        'updated_bank_df': updated_bank_df,
        'sum_updated_receipt': sum_updated_receipt,
        'sum_updated_bank': sum_updated_bank,
        'reconciliation': reconciliation,
        'reconciliation_status': 'succeeded' if sum_updated_receipt == sum_updated_bank else 'failed'
    }

