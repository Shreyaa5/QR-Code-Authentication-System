from data_preprocessing import show_samples
from train_ml import *
from cnn import *
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Fix UnicodeEncodeError: 'charmap' codec can't encode character

if __name__ == "__main__":
    print("🚀 Running QR Code Authentication Pipeline")
    show_samples()
    print("✅ ML & CNN Models Trained Successfully!")
