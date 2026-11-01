
from datasets import load_dataset
import pandas as pd
import os

print("Loading customer complaint dataset...")

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)

df = dataset.to_pandas()
print("\nDataset Downloaded!")
print("Dataset shape:",df.shape)

print("\nAvailable Columns:")
print(df.columns.tolist())

print(" \nFirst five records:")
print(df.head())

print("\nCategory distribution:")
print(df["category"].value_counts())

print("\nIntent distribution:")
print(df["intent"].value_counts())
os.makedirs("data/raw", exist_ok=True)

df.to_csv(
    "data/raw/customer_support.csv",
    index=False
)

print("\nDataset saved successfully!")
print("Location: data/raw/customer_support.csv")
