import os
import sys
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from tokenizer import get_tokenizer
from config import *

try:
     from tune import run_tune as _run_tune_module
except Exception:
     _run_tune_module = None

def run_tune():
    os.system("clear")
    os.system("python3 tune.py")

def run_generate():
    os.system("clear")
    os.system("python3 chat.py")

def run_train():
    os.system("clear")
    os.system("python3 train.py")

def run_config_edit():
    os.system("nano config.py")

def clean():
    os.system("clear")
    os.system("rm -r data/models training.log loss-resp.log progress.log tokenizer.model tokenizer.vocab data/training_data/_prebatched* data/training_data/_tokens.pt")

def main():
    os.system("clear")
    while True:
        print("\n=== LLM Project Main Menu ===")
        print("1. Generate text")
        print("2. Train model")
        print("3. Edit config")
        print("4. Clean data")
        print("5. Tuning")
        print("q. Quit")
        choice = input("Select an option: ").strip().lower()
        if choice == "1":
            run_generate()
        elif choice == "2":
            run_train()
        elif choice == "3":
            run_config_edit()
        elif choice == "4":
            clean()
        elif choice == "5":
            run_tune()
        elif choice == "q":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
