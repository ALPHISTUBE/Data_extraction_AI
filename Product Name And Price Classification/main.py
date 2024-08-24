import random
import pandas as pd
from data import products

# Define data categories and currency signs
categories = ['A', 'E', 'F', 'H', 'N', '£', '—']
currency_signs = [
    '$', '€', '£', '¥', '₹', '₽', '₩', '₫', '₪', '₭', '₮', '₦', '₲', '₵', '₸', '₺', '₡', '₢', '₱', '₴', '₯', 
    '฿', '₫', '₭', '៛', '₨', '৳', '৲', '₦', '₮', '₭', '₲', '₵', '₲', '₡', '₢', '₱', '₮', '₴', '₸', '₺', 
    '₦', '₲', '₵', '₸', '₯', '₳', '₭', '₢', '₱', '₴', '₯', '₨', '৳', '₡', '₲', '₯', '₮', '₢', '₱', '₴', 
    '₳', '₢', '₨', '৳', '៛', '₭', '₲', '₵', '₦', '₲', '₮', '₳', '₢', '₳', '₴', '₨', '₢', '₦', '₮'
]

_products = products

# Function to generate a product line entry
def generate_product_line(max_entries):
    entries = []
    for _ in range(max_entries):
        category = random.choice(categories)
        
        # Generate product code
        code_contains_letters = random.choice([True, False])
        if code_contains_letters:
            code = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))
        else:
            code = str(random.randint(100000, 999999))
        
        product = random.choice(_products)
        quantity = random.randint(1, 10)

        value_type = random.choice(['b', 's'])
        
        if value_type == "b":
            price = round(random.uniform(0.1, 99999.99), 2)
        else:
            price = round(random.uniform(0.1, 200.99), 2)
        
        # Decide whether to use a currency sign
        use_currency_sign = random.choice([True, False])
        if use_currency_sign:
            currency_sign = random.choice(currency_signs)
            priceS = f"{currency_sign}{price:,.2f}" if random.choice([True, False]) else f"{price:,.2f}{currency_sign}"
        else:
            priceS = f"{price:,.2f}"
        
        # Choose a format
        formats = [
            f"{code} {product} {priceS} {category}",
            f"{product} {priceS} {category}",
            f"{category} {product} {priceS}",
            f"{product} {priceS}",
            f"{product} {code} {priceS}",
        ]

        formats_quantity = [
            f"{product} {code} {quantity} {priceS}",
            f"{code} {product} {quantity} {priceS} {category}",
            f"{product} {quantity} {priceS} {category}",
            f"{category} {product} {quantity} {priceS}",
            f"{product} {quantity} {priceS}",
        ]

        use_format_quantity = random.choice([True, False])
        
        if use_format_quantity:
            line_format = random.choice(formats)
            entries.append([line_format, product, price])
        else:
            line_format = random.choice(formats_quantity)
            entries.append([line_format, product, quantity, price])

    return entries

# Tags for other line formats
tags = ['Total', 'subtotal', "sub-total", "st", "t", 'discount', 'cash', 'debit', 'credit', 'visa', 'master', 'change']

# Function to generate receipt lines with specific tags
def generate_receipt_line(max_entries):
    entries = []
    for _ in range(max_entries):
        line_type = random.choice(['tax', 'other'])  # Decide if the line is tax-related or other
        value_type = random.choice(['b', 's'])
        
        if value_type == "b":
            price = round(random.uniform(0.1, 99999.99), 2)
        else:
            price = round(random.uniform(0.1, 200.99), 2)
        
        # Decide whether to use a currency sign
        use_currency_sign = random.choice([True, False])
        if use_currency_sign:
            currency_sign = random.choice(currency_signs)
            priceS = f"{currency_sign}{price:,.2f}" if random.choice([True, False]) else f"{price:,.2f}{currency_sign}"
        else:
            priceS = f"{price:,.2f}"

        if line_type == 'tax':

            tax_type = random.choice(['tax', 'vat'])
            # Generate tax-specific line formats
            is_total_tax = random.choice([True, False])
            tax_serial = random.randint(0, 6)

            if value_type == "b":
                tax_rate = round(random.uniform(0.1, 100.0), 2)
            else:
                tax_rate = round(random.uniform(0.1, 10.0), 2)

            total_tax_formats = [
                f"total {tax_type} {priceS}",
                f"{priceS} total {tax_type}"
            ]

            tax_formats = [
                f"{tax_serial} {tax_type} {tax_rate}% {priceS}",
                f"{tax_serial} {tax_type} {tax_rate}% {priceS}",
                f"{tax_serial} {tax_rate}% {tax_type} {priceS}",
                f"{tax_type} {priceS}",
                f"{priceS} {tax_type}",
            ]
            
            if is_total_tax:
                line_format = random.choice(total_tax_formats)
                entries.append([line_format, f"total {tax_type}", price])
            else:
                line_format = random.choice(tax_formats)
                if line_format.find(f"{tax_rate}") != -1:
                    entries.append([line_format,f"{tax_type}", tax_rate, price])
                else:
                    entries.append([line_format,f"{tax_type}", price])
        else:
            # Generate other line formats
            tag = random.choice(tags)

            line_format = f"{tag} {priceS}"
            if tag == "st" or tag == "sub-total":
                tag = "subtotal"
            if tag == "t":
                tag = "total"

            entries.append([line_format, tag, price])

    return entries

# Example usage: Generate 10 product line entries
entries_pl = generate_product_line(100000)
entries_rl = generate_receipt_line(100000)

entries = entries_pl + entries_rl

# Convert to DataFrame and save to CSV
df = pd.DataFrame(entries)
df.to_csv('./receipt-line-dataset.csv', index=False)

print("Data has been written to receipt-line-dataset.csv")
