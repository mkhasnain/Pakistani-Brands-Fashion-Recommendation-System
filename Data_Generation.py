import random
import pandas as pd

# Define possible values for each feature
brands = ['Khaadi', 'Alkaram', 'Sana Safinaz', 'Bonanza', 'Satrangi', 'Gul Ahmed', 'Nishat Linen', 'Ego']
categories = ['Women\'s Wear', 'Kids\' Wear']
occasions = ['Casual', 'Formal', 'Party', 'Beach', 'Wedding', 'Mehendi', 'Walima', 'Eid']
color_themes = ['Green', 'Yellow', 'Red', 'Blue', 'N/A']  # N/A for occasions without specific color themes
styles = ['Dress', 'Top', 'Pants', 'Skirt']
colors = ['Yellow', 'Blue', 'Red', 'White', 'Green', 'Pink', 'Black', 'Purple']
patterns = ['Floral', 'Plain', 'Striped', 'Geometric', 'Polka Dot', 'Embroidered', 'Printed']
seasons = ['Summer', 'Winter']
materials = ['Cotton', 'Linen', 'Chiffon', 'Silk', 'Wool']
price_ranges = ['Low', 'Medium', 'High']
ratings = [round(random.uniform(3.0, 5.0), 1) for _ in range(100)]  # Random ratings between 3.0 and 5.0
popularities = ['Low', 'Medium', 'High']
availabilities = ['In Stock', 'Out of Stock']
discounts = ['0%', '5%', '10%', '15%', '20%', '25%']

# Generate synthetic data
data = []
for _ in range(100):  # Generate 100 rows of data
    occasion = random.choice(occasions)
    color_theme = random.choice(color_themes) if occasion == 'Mehendi' else 'N/A'
    row = {
        'Brand Name': random.choice(brands),
        'Category': random.choice(categories),
        'Occasion': occasion,
        'Color Theme': color_theme,
        'Style': random.choice(styles),
        'Color': random.choice(colors),
        'Pattern': random.choice(patterns),
        'Season': random.choice(seasons),
        'Material': random.choice(materials),
        'Price Range': random.choice(price_ranges),
        'Rating': random.choice(ratings),
        'Popularity': random.choice(popularities),
        'Availability': random.choice(availabilities),
        'Discount': random.choice(discounts)
    }
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('clothes_dataset.csv', index=False)

print("Enhanced dataset created and saved to 'clothes_dataset.csv'.")
