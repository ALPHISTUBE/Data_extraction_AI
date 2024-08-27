import pandas as pd
import re
import random
from datetime import datetime, timedelta

def generate_random_date():
    """Generate a random date in MM/DD format."""
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    return random_date.strftime('%m/%d')

def generate_random_price():
    """Generate a random price with a dollar sign."""
    return f"${round(random.uniform(5.0, 500.0), 2)}"  # Prices between $5 and $500

def replace_dates_and_prices(text):
    """Replace dates and prices in the text using regular expressions."""
    price_pattern = r'\$\d+(\.\d{2})?'
    date_pattern = r'\d{2}/\d{2}'
    
    text = re.sub(date_pattern, lambda match: generate_random_date(), text)
    text = re.sub(price_pattern, lambda match: generate_random_price(), text)
    
    return text

def generate_variations(description, num_variations):
    """Generate multiple variations of a description by changing dates and prices."""
    variations = []
    for _ in range(num_variations):
        new_description = replace_dates_and_prices(description)
        variations.append(new_description)
    return variations

def convert_csv(input_file, output_file, num_variations):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(input_file)
        
        # Check that the DataFrame has at least 3 columns
        if df.shape[1] < 3:
            raise ValueError("The input CSV does not have enough columns.")
        
        # Prepare lists to store the output data
        output_rows = []
        
        for _, row in df.iterrows():
            original_description = row['Transaction Message']
            if original_description != "Transaction Message":
                company = row['Company Name']
                category = row['Categories']
                
                variations = generate_variations(original_description, num_variations)
                
                for variation in variations:
                    output_rows.append([variation, f"('{company}', '{category}')"])
        
        # Create a new DataFrame for the output
        output_df = pd.DataFrame(output_rows, columns=['Transaction Message', 'Categories'])
        
        # Write the output DataFrame to a new CSV file
        output_df.to_csv(output_file, index=False, header=False)
        
        print(f"File converted and saved as '{output_file}'")
    
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
convert_csv('Transection_Description.csv', 'transaction-categories-dataset.csv', num_variations=20)
