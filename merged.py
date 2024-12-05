import requests
import json

# Base API URLs
products_url = 'https://accurateco2spares.com/api/products'
product_details_url = 'https://accurateco2spares.com/api/getProduct?id={}'

merged_data = []

# Fetch all products from the first API
products_response = requests.get(products_url)
products_data = products_response.json()

# Loop through each product to fetch additional details
#i=0
for product in products_data:
    product_id = product['id']
    print("currently done ",product_id)

    #i=i+1
    #if(i>10):
        #break

    # Fetch detailed product information from the second API using the product id
    details_response = requests.get(product_details_url.format(product_id))
    details_data = details_response.json()

    # Merge the basic product information with the detailed product information
    merged_product = { **details_data}

    # Add the merged product data to the final list
    merged_data.append(merged_product)

# Write merged data to a new JSON file
with open('merged_api.json', 'w') as f:
    json.dump(merged_data, f, indent=4)

print("API data has been successfully fetched and merged!")
