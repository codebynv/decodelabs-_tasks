import unittest
import sys
import os

# Add src to the path to import chatbot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import chatbot

class TestChatbot(unittest.TestCase):
    
    def test_greeting_recognition(self):
        self.assertEqual(chatbot.get_response("hello"), "Hi there! How can I assist you today?")
        self.assertEqual(chatbot.get_response("hi"), "Hello! How can I help?")
        
    def test_unknown_input_fallback(self):
        self.assertEqual(chatbot.get_response("something completely random"), "I do not understand. Try asking for 'help'.")
        self.assertEqual(chatbot.get_response(""), "I do not understand. Try asking for 'help'.")
        
    def test_input_normalization(self):
        # Testing uppercase and extra spaces
        self.assertEqual(chatbot.get_response("  HeLLO  "), "Hi there! How can I assist you today?")
        self.assertEqual(chatbot.get_response("WHAT CAN YOU DO?"), "I do not understand. Try asking for 'help'.") # No punctuation stripping implemented yet per spec, so this will fallback, which is deterministic.
        self.assertEqual(chatbot.get_response("  whaT can yoU dO  "), "I can respond to basic greetings and statements based on pre-defined rules.")
        
    def test_deterministic_behavior(self):
        # Multiple calls should yield the same response
        response1 = chatbot.get_response("who are you")
        response2 = chatbot.get_response("who are you")
        self.assertEqual(response1, response2)

if __name__ == '__main__':
    unittest.main()
