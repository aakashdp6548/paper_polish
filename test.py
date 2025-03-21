import os
import json

with open('iclr2025.json', 'r') as f:
    data = json.load(f)

print(len(data))


with open('iclr_reviews/2025.json', 'r') as f:
    reviews = json.load(f)

print(len(reviews))

print(f"icrl2025.json: {data[11676]['id']}")
print(f"Reviews.json: {reviews[11676]['id']}")
