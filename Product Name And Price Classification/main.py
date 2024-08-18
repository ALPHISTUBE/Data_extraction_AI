import csv
import random

# Define data categories
categories = ['A', 'E', 'F', 'H', 'N', '£', '—']
products = [
    'RED GRAPES', 'CORN CHIPS', 'WHOLE MILK', 'BROWN RICE CAKES', 'MANGO CHUNKS',
    'BROCCOLI FLORETS', 'DRIED PINEAPPLE', 'WHOLE WHEAT BREAD', 'SLICED SWISS CHEESE',
    'ORGANIC MILK', 'GRANOLA BARS', 'WHOLE WHEAT BREAD', 'BABY CARROTS', 'BEEF STEAK',
    'FROZEN FRUIT', 'WHOLE GRAIN CEREAL', 'COTTAGE CHEESE', 'COFFEE BEANS', 'SLICED PEPPERS',
    'OAT BRAN', 'ALMOND FLOUR', 'PASTA SAUCE', 'FRUIT JUICE', 'MUSHROOMS', 'GROUND TURKEY',
    'OREGANO', 'PUMPKIN PUREE', 'CINNAMON ROLLS', 'GREEN BEANS', 'YOGURT', 'BEEF TACOS',
    'CHERRY TOMATOES', 'GARLIC CLUMPS', 'ALMOND BUTTER', 'CARAMEL SAUCE', 'SPINACH',
    'DRIED FRUIT', 'WHOLE OATS', 'MULTIGRAIN BREAD', 'STRAWBERRY JAM', 'BAKING SODA',
    'APPLE JUICE', 'TOMATOES', 'CHEESE STICKS', 'SLICED CHICKEN', 'WHOLE CHICKEN',
    'PEAR SAUCE', 'FRUIT SNACKS', 'FROZEN YOGURT', 'SWEET CORN', 'CRACKERS', 'GARLIC POWDER',
    'SLICED MUSHROOMS', 'FROZEN CARROTS', 'FRENCH FRIES', 'PINEAPPLE CHUNKS', 'GREEK YOGURT',
    'SLICED TOMATOES', 'SLICED CHEESE', 'MARINARA SAUCE', 'SLICED APPLES', 'BEEF BURGERS',
    'WHOLE WHEAT PASTA', 'FROZEN BERRIES', 'TUNA FISH', 'SWEET POTATOES', 'CHOCOLATE SAUCE',
    'GREEN PEPPERS', 'FROZEN CHICKEN', 'SLICED JALAPENOS', 'WHOLE GRAIN BREAD', 'MANGOES',
    'GRANOLA', 'SMOKED SALMON', 'CHOPPED ONIONS', 'GROUND BEEF'
]

# Generate unique entries
entries = []
for i in range(200):
    category = random.choice(categories)
    code = random.randint(100000, 999999)
    product = random.choice(products)
    price = round(random.uniform(1.00, 12.99), 2)
    entries.append([category, code, product, price])

# Define the header
header = ['Code', 'Product Name', 'Price']

# Write to CSV file
with open('data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for entry in entries:
        writer.writerow([f'{category} {entry[1]} {entry[2]} {entry[3]:.2f} {entry[0]}', entry[2], f'{entry[3]:.2f}'])

print("Data has been written to data.csv")