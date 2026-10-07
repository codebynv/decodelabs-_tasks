# Project 1 — Rule-Based AI Chatbot

## Overview
This is a foundation-level Artificial Intelligence project built as part of the Decode Labs Industrial Training Kit (Batch 2026). It demonstrates a deterministic, rule-based chatbot constructed in Python without relying on external machine learning or generative libraries.

## Objective
The primary goal is to master control flow, deterministic logic (the "engineer" mindset), and continuous interaction loops before moving on to probabilistic models ("deep learning"). It acts as the logic engine that serves as a foundation for safe and structured AI systems.

## Requirements from Decode Labs
According to the project specification (IPO Model), the chatbot must:
1. Handle greetings and exit commands.
2. Rely on an explicit rule-based architecture for responses.
3. Use a continuous interaction loop (`while` cycle) until the user gives an exit command.
4. Normalize user input (handling case and whitespace).
5. Avoid "If-Elif Ladders" due to performance ($O(n)$) and maintenance overhead, favoring $O(1)$ Hash Maps/Dictionaries instead.
6. Provide a clean fallback response for unknown inputs.

## How It Works
The architecture utilizes the "IPO Model" (Input -> Process -> Output):
- **Sanitization & Normalization:** The input is stripped of surrounding whitespace and converted to lowercase.
- **Intent Matching (Knowledge Base):** The sanitized input acts as a key to look up predefined responses in a Python dictionary.
- **Response Generation:** The `.get()` method is used for an atomic operation that instantly retrieves the response or a default fallback if the input is not recognized.
- **Continuous Loop:** The application runs in an infinite loop that breaks only upon explicit kill commands (`exit`, `quit`, `bye`).

## Project Structure
```
project-01-rule-based-chatbot/
├── src/
│   └── chatbot.py            # Main application script
├── tests/
│   └── test_chatbot.py       # Unit tests for the bot's logic
├── .gitignore                # Git exclusion list
├── requirements.txt          # Project dependencies (empty as only stdlib is used)
└── README.md                 # Project documentation
```

## Technologies Used
- Python (Standard Library)
- `unittest` for automated testing

## How to Run
To run the chatbot, simply navigate to the project root and execute the source script:
```bash
python src/chatbot.py
```

## Example Interaction
```text
Welcome to the DecodeLabs Rule-Based AI Chatbot.
Type 'exit', 'quit', or 'bye' to end the conversation.

You: hello
Bot: Hi there! How can I assist you today?
You:   WHAT CAN YOU DO  
Bot: I can respond to basic greetings and statements based on pre-defined rules.
You: something completely random
Bot: I do not understand. Try asking for 'help'.
You: help
Bot: You can try saying hello, ask how I am, or type exit to quit.
You: exit
Bot: Goodbye!
```

## Testing
An automated test suite using `unittest` is provided to ensure correct intent matching, sanitization behavior, and fallback logic.

To run the tests:
```bash
python tests/test_chatbot.py
```
Tests passed: `4/4`

## What I Learned
- **Algorithmic Efficiency:** Transitioning from linear $O(n)$ if-elif ladders to constant-time $O(1)$ dictionary lookups significantly improves code maintainability and scalability.
- **Separation of Concerns:** Abstracting the response logic into `get_response()` separately from the I/O `while` loop makes the codebase easier to test deterministically.
- **Sanitization Importance:** Simple transformations like `.lower().strip()` are critical guardrails for rule-based systems to ensure reliable input matching.

## Limitations
- **Rigid Vocabulary:** The bot relies on exact string matching. "Hello!" (with punctuation) will fail if not handled explicitly.
- **No Context/Memory:** The bot processes each query independently and cannot remember previous conversation states.
- **Lack of Semantic Understanding:** It does not understand the meaning of words, only their hardcoded mappings.

## Future Improvements
- Expand the knowledge base with more intents.
- Implement regex or basic NLP tokenization to handle punctuation and partial matches.
- Introduce session memory to maintain conversational context.
