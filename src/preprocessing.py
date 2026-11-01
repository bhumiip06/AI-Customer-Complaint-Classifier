import pandas as pd
import re
import os

def clean_text(text):
    text=str(text).lower()

    #remove urls
    text= re.sub(r"http\S+","",text)
    #remove special characters but preserve numbers
    text=re.sub(r"[^a-z0-9\s]",'',text)
    #remove extra spaces
    text=re.sub(r"\s+"," ",text)

    return text.strip()

def preprocess_dataset():
    print("loading dataset...")

    df=pd.read_csv("data/raw/customer_support.csv")

    print("Original shape:",df.shape)

    df=df[["instruction", "category"]].copy()

    #remove missing values
    df.dropna()

    #clean text and labels
    df["instruction"]=df["instruction"].apply(clean_text)

    df["category"]=(df["category"].astype(str).str.strip())

    # Remove empty records
    df = df[
        (df["instruction"] != "") &
        (df["category"] != "")
    ]

    # Remove duplicate records
    df.drop_duplicates(
        subset=["instruction", "category"]
    )

    # Save processed dataset
    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(
        "data/processed/cleaned_complaints.csv",
        index=False
    )

    print("\nPreprocessing completed!")

    print("Final shape:", df.shape)

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    print("\nSample processed records:")
    print(df.head())

    return df


if __name__ == "__main__":
    preprocess_dataset()
