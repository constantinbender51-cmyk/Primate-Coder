import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import string
import random

# Generate character dataset with em dashes
def generate_character_dataset(n_samples=5000, em_dash_ratio=0.2):
    """Generate a dataset of characters with em dashes mixed in"""
    characters = []
    labels = []
    
    # Regular ASCII characters
    ascii_chars = string.ascii_letters + string.digits + string.punctuation + ' '
    
    # Generate dataset
    n_em_dashes = int(n_samples * em_dash_ratio)
    n_other = n_samples - n_em_dashes
    
    # Add em dashes
    for _ in range(n_em_dashes):
        characters.append('—')  # Em dash character
        labels.append('Em Dash')
    
    # Add other characters
    for _ in range(n_other):
        char = random.choice(ascii_chars)
        characters.append(char)
        labels.append('Other')
    
    # Shuffle the dataset
    combined = list(zip(characters, labels))
    random.shuffle(combined)
    characters, labels = zip(*combined)
    
    return list(characters), list(labels)

def extract_features(characters):
    """Extract features from characters for ML model"""
    features = []
    
    for i, char in enumerate(characters):
        # Feature 1: ASCII/Unicode value
        ascii_val = ord(char) if char else 0
        
        # Feature 2: Character type flags
        is_punctuation = 1 if char in string.punctuation else 0
        is_letter = 1 if char.isalpha() else 0
        is_digit = 1 if char.isdigit() else 0
        is_space = 1 if char.isspace() else 0
        
        # Feature 3: Is em dash (target characteristic)
        is_em_dash_like = 1 if char == '—' else 0
        
        # Feature 4: Character length (always 1 for single chars)
        char_length = len(char)
        
        # Feature 5: Position in sequence
        position = i
        
        # Feature 6: Unicode category
        unicode_cat = ord(char) // 100  # Simplified category
        
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
    print("\nGenerating character dataset...")
    characters, labels = generate_character_dataset()
    
    # Count classes
    em_dash_count = labels.count('Em Dash')
    other_count = labels.count('Other')
    print(f"Total characters: {len(characters)}")
    print(f"Em dashes: {em_dash_count}")
    print(f"Other characters: {other_count}")
    
    # Extract features
    print("\nExtracting features...")
    X = extract_features(characters)
    y = np.array([1 if label == 'Em Dash' else 0 for label in labels])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Features per sample: {X.shape[1]}")
    
    # Train model
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
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
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Other', 'Em Dash']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    print("\nFeature Importance:")
    feature_names = [
        'ASCII Value', 'Is Punctuation', 'Is Letter', 'Is Digit', 
        'Is Space', 'Is Em-dash-like', 'Char Length', 'Position', 'Unicode Category'
    ]
    importances = model.feature_importances_
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.4f}")
    
    # Sample predictions
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)
    
    test_chars = ['a', '—', '1', '.', '—', 'Z', ' ']
    for char in test_chars:
        features = extract_features([char])
        proba = model.predict_proba(features)[0]
        prediction = 'Em Dash' if model.predict(features)[0] == 1 else 'Other'
        confidence = proba[1] if prediction == 'Em Dash' else proba[0]
        print(f"Character '{char}' -> {prediction} (confidence: {confidence:.4f})")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

# Run the training
if __name__ == "__main__":
    train_em_dash_detector()