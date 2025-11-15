import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Generate the same dataset as in script.py
def generate_dataset():
    np.random.seed(42)  # For reproducibility
    
    # Total characters and em dashes
    total_chars = 5000
    em_dash_count = 1000
    
    # Create character pool (printable ASCII + em dash)
    chars = []
    for i in range(32, 127):  # Printable ASCII
        chars.append(chr(i))
    
    # Generate random characters
    all_chars = []
    for _ in range(total_chars - em_dash_count):
        all_chars.append(np.random.choice(chars))
    
    # Add em dashes
    for _ in range(em_dash_count):
        all_chars.append('—')  # em dash
    
    # Shuffle the characters
    np.random.shuffle(all_chars)
    
    return all_chars

def create_features_labels(characters):
    """Convert characters to features and labels for ML"""
    
    # Encode characters as features
    le = LabelEncoder()
    
    # Create features: character encoding and positional features
    features = []
    labels = []
    
    for i, char in enumerate(characters):
        # Feature 1: ASCII value (or encoded value)
        if char == '—':
            char_encoded = 1000  # Special value for em dash
        else:
            char_encoded = ord(char)
        
        # Feature 2: Position in the sequence
        position = i
        
        # Feature 3: Row and column position (for 10x500 matrix)
        row = i // 500
        col = i % 500
        
        # Feature 4: Is it a special character? (punctuation, etc.)
        is_punctuation = 1 if char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' else 0
        
        # Feature 5: Is it alphanumeric?
        is_alphanumeric = 1 if char.isalnum() else 0
        
        # Feature 6: Is it whitespace?
        is_whitespace = 1 if char.isspace() else 0
        
        features.append([char_encoded, position, row, col, is_punctuation, is_alphanumeric, is_whitespace])
        labels.append(1 if char == '—' else 0)
    
    return np.array(features), np.array(labels), le

def train_model():
    print("Generating dataset...")
    characters = generate_dataset()
    
    print("Creating features and labels...")
    X, y, label_encoder = create_features_labels(characters)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of em dashes: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train Random Forest classifier
    print("\nTraining Random Forest classifier...")
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Em Dash', 'Em Dash']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    print("\nFeature Importances:")
    feature_names = ['Char Encoded', 'Position', 'Row', 'Col', 'Is Punctuation', 'Is Alphanumeric', 'Is Whitespace']
    for name, importance in zip(feature_names, rf_classifier.feature_importances_):
        print(f"{name}: {importance:.4f}")
    
    # Save the model
    joblib.dump(rf_classifier, 'em_dash_classifier.pkl')
    print("\nModel saved as 'em_dash_classifier.pkl'")
    
    # Test on a few examples
    print("\nTesting on sample characters:")
    test_chars = ['A', '—', '!', '—', 'z', ' ']
    for char in test_chars:
        if char == '—':
            char_encoded = 1000
        else:
            char_encoded = ord(char)
        
        # Create a dummy feature vector (position 0, row 0, col 0)
        features = np.array([[char_encoded, 0, 0, 0, 
                             1 if char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' else 0,
                             1 if char.isalnum() else 0,
                             1 if char.isspace() else 0]])
        
        prediction = rf_classifier.predict(features)[0]
        probability = rf_classifier.predict_proba(features)[0]
        
        print(f"Character '{char}': Predicted = {'Em Dash' if prediction == 1 else 'Not Em Dash'}, "
              f"Probability = {probability[1]:.4f}")

if __name__ == "__main__":
    train_model()