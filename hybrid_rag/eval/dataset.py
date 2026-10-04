"""Golden Q/A set for Ragas eval."""
GOLDEN = [
    {"question": "How many vacation days do employees get?",
     "ground_truth": "Employees get 20 paid vacation days per year.",
     "contexts": ["Employees receive 20 paid vacation days annually."]},
    {"question": "What is the refund policy?",
     "ground_truth": "Refunds are available within 30 days of purchase.",
     "contexts": ["Refunds available within 30 days of purchase with receipt."]},
    {"question": "How do I reset my password?",
     "ground_truth": "Use the forgot-password link on the login page.",
     "contexts": ["Reset via the forgot-password link on the login page."]},
]
