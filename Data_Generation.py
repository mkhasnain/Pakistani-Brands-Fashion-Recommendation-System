import random
import pandas as pd
# Define possible values for each feature
brands = ['Khaadi', 'Alkaram', 'Sana Safinaz', 'Bonanza', 'Satrangi', 'Gul Ahmed']
categories = ['Women\'s Wear', 'Kids\' Wear']
occasions = ['Casual', 'Formal', 'Party', 'Beach', 'Wedding', 'Mehendi', 'Walim
color_themes = ['Green', 'Yellow', 'Red', 'Blue', 'N/A'] # N/A for occasions w
styles = ['Dress', 'Top', 'Pants', 'Skirt']
colors = ['Yellow', 'Blue', 'Red', 'White', 'Green', 'Pink', 'Black', 'Purple']
patterns = ['Floral', 'Plain', 'Striped', 'Geometric', 'Polka Dot', 'Embroidere
seasons = ['Summer', 'Winter']
materials = ['Cotton', 'Linen', 'Chiffon', 'Silk', 'Wool']
price_ranges = ['Low', 'Medium', 'High']
ratings = [round(random.uniform(3.0, 5.0), 1) for _ in range(100)] # Random ra
popularities = ['Low', 'Medium', 'High']
availabilities = ['In Stock', 'Out of Stock']
discounts = ['0%', '5%', '10%', '15%', '20%', '25%']
