import requests
import json

# Base API URLs
products_url = 'https://accurateco2spares.com/api/products'
subcategory_url_template = 'https://accurateco2spares.com/api/getProduct?id={}'

merged_data = {
    "products": [],
    "categories": {}
}

try:
    # Fetch all products
    response = requests.get(products_url)
    response.raise_for_status()
    products_data = response.json()

    # Limit the number of products to 10 for testing
    test_limit = 10
    products_to_process = products_data[:test_limit]

    # Loop through each product to fetch subcategory details
    for product in products_to_process:
        subcategory_id = product.get('subcategory_id')  # Assuming 'subcategory_id' is the key in the product data
        if subcategory_id:
            subcategory_url = subcategory_url_template.format(subcategory_id)
            subcategory_response = requests.get(subcategory_url)
            subcategory_response.raise_for_status()
            subcategory_data = subcategory_response.json()

            # Merge product data with its subcategory details
            product['subcategory_details'] = subcategory_data

            # Add subcategory details to merged data if not already present
            if subcategory_id not in merged_data['categories']:
                merged_data['categories'][subcategory_id] = subcategory_data

        # Append the product with its detailed subcategory to the products list
        merged_data["products"].append(product)

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except Exception as err:
    print(f"Other error occurred: {err}")

# Write merged data to a new JSON file
with open('merged_api.json', 'w') as f:
    json.dump(merged_data, f, indent=4)

print("API data for 10 products has been successfully fetched and merged!")
