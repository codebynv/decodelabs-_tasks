def get_response(user_input):
    """
    Returns the appropriate response for the given user input based on a dictionary knowledge base.
    """
    # The PDF emphasizes using a dictionary for $O(1)$ lookup over an if-elif ladder.
    knowledge_base = {
        'hello': 'Hi there! How can I assist you today?',
        'hi': 'Hello! How can I help?',
        'hey': 'Hey! What can I do for you?',
        'how are you': 'I am a simple rule-based bot, but I am doing well! How about you?',
        'what can you do': 'I can respond to basic greetings and statements based on pre-defined rules.',
        'who are you': 'I am an AI Chatbot created for Project 1 of Decode Labs.',
        'help': 'You can try saying hello, ask how I am, or type exit to quit.'
    }
    
    # Sanitization: Handle case & whitespace
    clean_input = user_input.lower().strip()
    
    # Implementation: The .get() method for atomic lookup + fallback
    return knowledge_base.get(clean_input, "I do not understand. Try asking for 'help'.")

def main():
    print("Welcome to the DecodeLabs Rule-Based AI Chatbot.")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.\n")
    
    exit_commands = {'exit', 'quit', 'bye'}
    
    # Input Loop: Continuous 'while' cycle
    while True:
        try:
            raw_input = input("You: ")
            clean_input = raw_input.lower().strip()
            
            if clean_input in exit_commands:
                print("Bot: Goodbye!")
                break
            
            response = get_response(clean_input)
            print(f"Bot: {response}")
            
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Exiting unexpectedly. Goodbye!")
            break

if __name__ == "__main__":
    main()
