import csv
import pandas as pd
csv_file_path = '/media/alphi/Alphi/Python/Python Project/Spensibily_OCR/Product_Name_And_Price_Classification/Dataset/Bank/transaction_480k+.csv'

categories_list = [
    "Supermarkets and Groceries",
    "Healthcare",
    "Internal Account Transfer",
    "Payroll",
    "Third Party",
    "Department Stores",
    "Insurance",
    "Loans",
    "Convenience Stores",
    "Gas Stations",
    "Transfer Debit",
    "Travel",
    "Clothing and Accessories",
    "Shops",
    "Utilities",
    "Check Deposit",
    "Food and Beverage Services",
    "ATM",
    "Telecommunication Services",
    "Transfer Deposit",
    "Payment",
    "Tax Refund",
    "Gyms and Fitness Centers",
    "Interest",
    "Digital Entertainment",
    "Transfer Credit",
    "Service",
    "Bank Fees",
    "Arts and Entertainment",
    "Restaurants"
]


CATEGORIES = [
            "Rent",
            "Utilities (Electricity)",
            "Utilities (Water)",
            "Utilities (Gas)",
            "Internet Service",
            "Telephone Service",
            "POS System Fees",
            "Inventory Purchases",
            "Shipping & Freight Costs",
            "Packaging Supplies",
            "Store Supplies",
            "Cleaning Supplies",
            "Marketing & Advertising",
            "Website Maintenance",
            "E-commerce Platform Fees",
            "Merchant Processing Fees",
            "Payroll",
            "Employee Benefits",
            "Employee Training",
            "Uniforms",
            "Security Services",
            "Insurance (Property)",
            "Insurance (Liability)",
            "Insurance (Workers' Compensation)",
            "Insurance (Health)",
            "Taxes (Sales)",
            "Taxes (Property)",
            "Taxes (Payroll)",
            "Accounting Services",
            "Legal Services",
            "Professional Fees",
            "Banking Fees",
            "Interest Expenses",
            "Loan Repayments",
            "Equipment Leasing",
            "Equipment Repairs",
            "Store Decorations",
            "Office Supplies",
            "Travel Expenses",
            "Vehicle Leasing",
            "Vehicle Maintenance",
            "Fuel Costs",
            "Delivery Expenses",
            "Customer Returns",
            "Discounts Given",
            "Employee Discounts",
            "Refunds Issued",
            "Loyalty Program Costs",
            "Gift Wrapping Supplies",
            "Gift Card Processing Fees",
            "Bad Debts",
            "Depreciation",
            "Amortization",
            "Charitable Donations",
            "Community Sponsorships",
            "Staff Meals",
            "Holiday Decorations",
            "Seasonal Displays",
            "Window Display Costs",
            "Signage Costs",
            "Music Licensing Fees",
            "Pest Control",
            "Landscaping",
            "Snow Removal",
            "Waste Disposal",
            "Recycling Services",
            "Warehouse Rent",
            "Warehouse Supplies",
            "Warehouse Utilities",
            "Cash Register Supplies",
            "ATM Fees",
            "Credit Card Fees",
            "Debit Card Fees",
            "Employee Recruitment",
            "Employee Onboarding",
            "Training Materials",
            "Conferences & Seminars",
            "Subscriptions & Dues",
            "Membership Fees",
            "IT Support Services",
            "Software Licenses",
            "Cloud Storage Services",
            "Data Backup Services",
            "Printing & Copying",
            "Office Equipment",
            "Mail & Courier Services",
            "Customer Service Expenses",
            "Inventory Shrinkage",
            "Product Sampling Costs",
            "Store Fixtures",
            "In-store Events",
            "Promotional Materials",
            "Survey & Feedback Tools",
            "Business Cards",
            "Loyalty Cards",
            "Business Licenses & Permits",
            "Environmental Fees",
            "Employee Wellness Programs"
        ]

def count_unique_categories(csv_file_path):
    categories = set()
    
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            category = row.get('category')
            if category:
                categories.add(category)
    
    print(f"Number of unique categories: {len(categories)}")
    print("Categories:")
    for category in categories:
        print(category)

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
    [0, "income"],
    [1, "expense"],
    [2, "unrecognized"],
    [0, "Income"],
    [1, "Expense"],
    [2, "Unrecognized"],
    [0, "0"],
    [1, "1"],
    [2, "2"]
]

# Load CSV, replace, and save
def replace_category_in_csv(input_file, output_file):
    with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["purpose"]  # Add new header "purpose"
        
        # If the "category" column exists
        if "category" in fieldnames:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                original_category = row["category"]
                transaction_type = row["transaction_type"]
                
                # Check if the category matches the first item in any of the similar_categories
                for transaction_pair in transaction_types:
                    if transaction_type == transaction_pair[1]:
                        row["transaction_type"] = transaction_pair[0]
                        break

                for category_pair in similar_category:
                    if original_category == category_pair[1]:
                        row["category"] = category_pair[0]  # Replace with the second item
                        for purpose_pair in payment_categories:
                            if category_pair[3] == purpose_pair[1]:
                                row["purpose"] = purpose_pair[0]
                        break
                else:
                    row["purpose"] = ""  # Default to empty if no match
                
                # Write updated row to new CSV
                writer.writerow(row)

        

# Replace categories and save to a new CSV
replace_category_in_csv(csv_file_path, '/media/alphi/Alphi/Python/Python Project/Spensibily_OCR/Product_Name_And_Price_Classification/Dataset/Bank/Transaction-Data/all/output_file_480k.csv')

# Read the CSV and print unique categories from the "category" header
def print_unique_categories(csv_file_path):
    df = pd.read_csv(csv_file_path)
    unique_categories = df['transaction_type'].unique()
    unique_categories_list = list(unique_categories)
    index = 0
    for category in unique_categories_list:
        print(category)


# Call the function to print unique categories
print_unique_categories("/media/alphi/Alphi/Python/Python Project/Spensibily_OCR/Product_Name_And_Price_Classification/Dataset/Bank/Transaction-Data/all/output_file_480k.csv")
