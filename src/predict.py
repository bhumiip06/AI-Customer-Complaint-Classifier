import joblib
# Load trained model
print("Loading trained model...")

model = joblib.load("models/best_model.pkl")

print("Model loaded successfully!")

# Function to predict complaint category
def predict_complaint(complaint):

    prediction = model.predict([complaint])

    return prediction[0]


# Test with sample complaints
if __name__ == "__main__":

    print("\nAI CUSTOMER COMPLAINT CLASSIFIER")
    print("--------------------------------")

    while True:

        complaint = input("\nEnter customer complaint (or type 'exit'): ")

        if complaint.lower() == "exit":
            print("Thank you for using the classifier!")
            break

        if not complaint.strip():
            print("Please enter a valid complaint.")
            continue

        category = predict_complaint(complaint)

        print("\nPredicted Complaint Category:", category)
