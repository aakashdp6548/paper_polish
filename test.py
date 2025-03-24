import os
import json

with open('iclr/paperlists/iclr2024.json', 'r') as f:
    data = json.load(f)

print(len(data))


with open('iclr/reviews/2024.json', 'r') as f:
    reviews = json.load(f)

print(len(reviews))

print(f"icrl2025.json: {data[7406]['id']}")
print(f"Reviews.json: {reviews[7406]['id']}")
