# Chatbot implementation with PyTorch
This Chatbot is currently working with Machine learning algorithms. Decisions are made by Neural Network algorithm.
Where only Forward Propagation occurs with two hidden layers. This process is the way to move from the Input layer (left) to the Output layer (right) in the neural network.
Easily customize your questions and answers by modifying QA.json with possible patterns and responses and re-run the training.
# Installation:
Create and activate a virtual environment(e.g. conda or venv)
Add an __init__ file to your project folder, so that it can be recognized and treated as a python project.
Install PyTorch, nltk and dependencies
# Usage:
Run python train.py
This will dump data.pth file and then run python chatbot_main.py.
Whenever you **modify** the json file. You have rerun the train.py and chatbot_main.py.

{
    "intents": [
      {
        "tag": "greeting",
        "patterns": [
          "Hi",
          "Hey",
          "How are you",
          "Is anyone there?",
          "Hello",
          "Good day"
        ],
        "responses": [
          "Hey :-)",
          "Hello, thanks for visiting our site",
          "Hi there, what can I do for you?",
          "Hi there, how can I help?"
        ]
      },
      {
        "tag": "goodbye",
        "patterns": ["Bye", "Thanks for your help", "Goodbye"],
        "responses": [
          "Expecting your order, thanks for checking us out",
          "Have a nice day",
          "Bye! Come back again soon."
        ]
       }
