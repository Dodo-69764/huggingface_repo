from transformers import pipeline

# Load default sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

# Test a few inputs
results = classifier(["I love this internship!", "This code is a disaster."])
for result in results:
    print(f"Label: {result['label']}, Score: {result['score']:.2f}")