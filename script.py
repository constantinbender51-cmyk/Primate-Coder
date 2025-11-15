import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import string
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_character_dataset():
    """Generate a dataset of 5000 characters with 1000 hidden em dashes"""
    print("Generating character dataset...")
    
    # Total characters and em dashes
    total_chars = 5000
    num_em_dashes = 1000
    
    # Create a list to store characters and labels
    characters = []
    labels = []  # 1 for em dash, 0 for other
    
    # Add em dashes
    for _ in range(num_em_dashes):
        characters.append('—')  # em dash
        labels.append(1)
    
    # Add other characters (mix of letters, numbers, punctuation)
    for _ in range(total_chars - num_em_dashes):
        # Mix of different character types
        char_type = random.choice(['letter', 'digit', 'punctuation', 'space'])
        
        if char_type == 'letter':
            char = random.choice(string.ascii_letters)
        elif char_type == 'digit':
            char = random.choice(string.digits)
        elif char_type == 'punctuation':
            char = random.choice(string.punctuation.replace('—', ''))
        else:
            char = ' '
        
        characters.append(char)
        labels.append(0)
    
    # Shuffle the dataset
    combined = list(zip(characters, labels))
    random.shuffle(combined)
    characters, labels = zip(*combined)
    
    print(f"Total characters: {len(characters)}")
    print(f"Em dashes: {sum(labels)}")
    print(f"Other characters: {len(labels) - sum(labels)}")
    
    return list(characters), list(labels)

def extract_features(characters):
    """Extract features from characters for machine learning"""
    features = []
    
    for i, char in enumerate(characters):
        # Feature 1: ASCII value (or Unicode code point)
        ascii_val = ord(char)
        
        # Feature 2: Character type indicators
        is_punctuation = 1 if char in string.punctuation or char == '—' else 0
        is_letter = 1 if char in string.ascii_letters else 0
        is_digit = 1 if char in string.digits else 0
        is_space = 1 if char == ' ' else 0
        is_em_dash_like = 1 if char in ['—', '-', '–'] else 0  # em dash, hyphen, en dash
        
        # Feature 3: String length (always 1 for single characters)
        char_length = len(char)
        
        # Feature 4: Position in sequence (normalized)
        position = i / len(characters)
        
        # Feature 5: Unicode category (simplified)
        if char in string.ascii_lowercase:
            unicode_cat = 1
        elif char in string.ascii_uppercase:
            unicode_cat = 2
        elif char in string.digits:
            unicode_cat = 3
        elif char in string.punctuation:
            unicode_cat = 4
        elif char == ' ':
            unicode_cat = 5
        elif char == '—':
            unicode_cat = 6
        else:
            unicode_cat = 0
        
        features.append([
            ascii_val,
            is_punctuation,
            is_letter,
            is_digit,
            is_space,
            is_em_dash_like,
            char_length,
            position,
            unicode_cat
        ])
    
    return np.array(features)

def train_em_dash_detector():
    """Main function to train the em dash detection model"""
    print("=" * 80)
    print("MACHINE LEARNING EM DASH DETECTOR")
    print("=" * 80)
    
    # Generate dataset
    characters, labels = generate_character_dataset()
    
    # Extract features
    print("\nExtracting features...")
    X = extract_features(characters)
    y = np.array(labels)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Features per sample: {X_train.shape[1]}")
    
    # Train Random Forest classifier
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Other', 'Em Dash']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    print("\nFeature Importance:")
    feature_names = [
        'ASCII Value', 'Is Punctuation', 'Is Letter', 'Is Digit', 'Is Space',
        'Is Em-dash-like', 'Char Length', 'Position', 'Unicode Category'
    ]
    
    importances = model.feature_importances_
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.4f}")
    
    # Test on some examples
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)
    
    test_chars = ['a', '—', '1', '.', '—', 'Z', ' ']
    test_features = extract_features(test_chars)
    predictions = model.predict(test_features)
    prediction_probs = model.predict_proba(test_features)
    
    for char, pred, prob in zip(test_chars, predictions, prediction_probs):
        pred_label = 'Em Dash' if pred == 1 else 'Other'
        confidence = prob[1] if pred == 1 else prob[0]
        print(f"Character '{char}' -> {pred_label} (confidence: {confidence:.4f})")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    train_em_dash_detector()