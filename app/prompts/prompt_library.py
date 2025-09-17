from langchain_core.prompts import ChatPromptTemplate

# Baseline (Zero-shot) prompt for text classification
baseline_classification_prompt = ChatPromptTemplate.from_template("""
You are a content moderation system. Classify the following text into exactly one of these categories: toxic, spam, or safe.

Text to classify: {text}

Respond with only one word: toxic, spam, or safe.

Classification:
""")

# Advanced (Few-shot + Role-based) prompt for text classification
advanced_classification_prompt = ChatPromptTemplate.from_template("""
You are an expert content moderator with years of experience in online safety. Your job is to classify user-generated content to maintain a healthy community environment.

Classification Guidelines:
- TOXIC: Content that is harmful, abusive, hateful, threatening, or promotes violence
- SPAM: Promotional content, repetitive messages, phishing attempts, or commercial solicitation
- SAFE: Normal, appropriate content that doesn't violate community standards

Examples:
Text: "Great article! Thanks for sharing this helpful information."
Classification: safe

Text: "You're worthless and nobody likes you. Kill yourself."
Classification: toxic

Text: "🚨 URGENT! Make $5000/week working from home! Click this link now! Limited time offer!!!"
Classification: spam

Text: "I disagree with your opinion, but I respect your right to have it."
Classification: safe

Text: "F*ck you, you stupid piece of sh*t!"
Classification: toxic

Text: "Buy cheap viagra now! No prescription needed! www.sketchy-pills.com"
Classification: spam

Now classify this text:
Text: {text}

Respond with only one word: toxic, spam, or safe.

Classification:
""")

# Central dictionary to register prompts
PROMPT_REGISTRY = {
    "baseline_classification": baseline_classification_prompt,
    "advanced_classification": advanced_classification_prompt,
}