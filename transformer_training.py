import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import io
import matplotlib.pyplot as plt
import gdown
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class PositionalEncoding(layers.Layer):
    """Fixed positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix using numpy for simplicity
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((max_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

class MultiHeadAttention(layers.Layer):
    """Multi-head attention mechanism"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.dense(concat_attention), attention_weights

class TransformerEncoderLayer(layers.Layer):
    """Single transformer encoder layer"""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=None, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class TransformerEncoder(layers.Layer):
    """Complete transformer encoder"""
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(d_model, maximum_position_encoding)
        
        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
                          for _ in range(num_layers)]
        
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training=None, mask=None):
        seq_len = tf.shape(x)[1]
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        
        return x

def download_ohlcv_data():
    """Download OHLCV data from Google Drive using gdown"""
    print("Downloading OHLCV data from Google Drive...")
    
    # Google Drive file ID from the URL
    file_id = "1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o"
    
    # Output filename
    output_file = "ohlcv_data.csv"
    
    try:
        # Download using gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_file, quiet=False)
        
        # Load the downloaded CSV
        data = pd.read_csv(output_file)
        print(f"Successfully downloaded data with shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        
        return data
    
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Creating sample OHLCV data for demonstration...")
        
        # Create sample data if download fails
        dates = pd.date_range('2018-01-01', '2018-12-31', freq='1min')
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        price = 100.0
        data = []
        
        for date in dates:
            # Random walk for price
            change = np.random.normal(0, 0.001)
            price = price * (1 + change)
            
            # Generate OHLC from the price
            open_price = price
            high_price = price * (1 + abs(np.random.normal(0, 0.002)))
            low_price = price * (1 - abs(np.random.normal(0, 0.002)))
            close_price = price
            volume = np.random.randint(1000, 10000)
            
            data.append([date, open_price, high_price, low_price, close_price, volume])
        
        return pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

def preprocess_data(data):
    """Preprocess OHLCV data for transformer training"""
    print("Preprocessing data...")
    
    # Check if we have the expected columns, if not try to find them
    expected_columns = ['open', 'high', 'low', 'close', 'volume']
    available_columns = data.columns.tolist()
    
    print(f"Available columns: {available_columns}")
    
    # Try to find close price column
    close_col = None
    for col in available_columns:
        if 'close' in col.lower():
            close_col = col
            break
    
    if close_col is None:
        # If no close column found, use the last numeric column
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            close_col = numeric_cols[-1]
        else:
            raise ValueError("No suitable close price column found")
    
    # Calculate technical indicators
    data['returns'] = data[close_col].pct_change()
    data['volatility'] = data['returns'].rolling(window=20).std()
    data['sma_20'] = data[close_col].rolling(window=20).mean()
    data['sma_50'] = data[close_col].rolling(window=50).mean()
    
    # Find volume column
    volume_col = None
    for col in available_columns:
        if 'volume' in col.lower():
            volume_col = col
            break
    
    if volume_col:
        data['volume_sma'] = data[volume_col].rolling(window=20).mean()
        feature_columns = [close_col, volume_col, 'returns', 'volatility', 'sma_20', 'sma_50', 'volume_sma']
    else:
        feature_columns = [close_col, 'returns', 'volatility', 'sma_20', 'sma_50']
    
    # Remove NaN values
    data = data.dropna()
    
    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[feature_columns])
    
    return scaled_features, scaler, feature_columns

def create_sequences_memory_efficient(data, sequence_length=60, prediction_horizon=5, max_sequences=50000):
    """Create sequences with memory limits to prevent SIGKILL errors"""
    X, y = [], []
    
    # Calculate how many sequences we can create
    total_possible = len(data) - sequence_length - prediction_horizon
    sample_rate = max(1, total_possible // max_sequences)
    
    print(f"Total possible sequences: {total_possible:,}")
    print(f"Sampling 1 in every {sample_rate} sequences to limit memory usage")
    print(f"Creating maximum of {max_sequences:,} sequences")
    
    for i in range(0, total_possible, sample_rate):
        if len(X) >= max_sequences:
            break
            
        # Input sequence
        X.append(data[i:(i + sequence_length)])
        
        # Target: future price movement (1 if price goes up, 0 if down)
        # Close price is typically the first column after preprocessing
        current_close = data[i + sequence_length - 1, 0]  # First column is close price
        future_close = data[i + sequence_length + prediction_horizon - 1, 0]
        
        # Binary classification: 1 if price increases, 0 if decreases
        target = 1 if future_close > current_close else 0
        y.append(target)
    
    print(f"Successfully created {len(X):,} sequences")
    return np.array(X), np.array(y)

def build_transformer_model(input_shape, num_layers=4, d_model=256, num_heads=16, 
                          dff=1024, dropout_rate=0.1):
    """Build the transformer model for time series classification"""
    
    inputs = keras.Input(shape=input_shape)
    
    # Transformer encoder
    transformer_encoder = TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_shape[1],
        maximum_position_encoding=1000,
        dropout_rate=dropout_rate
    )
    
    # Pass through transformer
    encoder_output = transformer_encoder(inputs, training=True)
    
    # Global average pooling and classification with better initialization
    x = layers.GlobalAveragePooling1D()(encoder_output)
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(x)
    outputs = layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def train_transformer():
    """Main training function"""
    print("Starting Transformer Training for OHLCV Data...")
    
    # Download and preprocess data
    data = download_ohlcv_data()
    scaled_features, scaler, feature_columns = preprocess_data(data)
    
    print(f"Processed data shape: {scaled_features.shape}")
    print(f"Features used: {feature_columns}")
    
    # Create sequences
    sequence_length = 60  # 60 minutes of historical data
    prediction_horizon = 5  # Predict 5 minutes ahead
    
    X, y = create_sequences_memory_efficient(scaled_features, sequence_length, prediction_horizon, max_sequences=20000)
    
    print(f"Sequences created: {X.shape}")
    print(f"Target distribution: {np.unique(y, return_counts=True)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build model
    # Build model
    model = build_transformer_model(
        input_shape=(sequence_length, len(feature_columns)),
        num_layers=4,
        d_model=256,
        num_heads=16,
        dff=1024
    )
    
    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    # Compile model with better optimizer settings
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train model with class weights and callbacks
    print("\nTraining transformer model...")
    
    # Add learning rate scheduler
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=5,
        validation_data=(X_test, y_test),
        verbose=2,
        class_weight=class_weight_dict,
        callbacks=[lr_scheduler]
    )
    
    # Evaluate model with custom threshold
    print("\nEvaluating model...")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    # Also calculate metrics with custom threshold
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.4).astype(int).flatten()  # Lower threshold
    
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    custom_precision = precision_score(y_test, y_pred, zero_division=0)
    custom_recall = recall_score(y_test, y_pred, zero_division=0)
    custom_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy (default): {test_accuracy:.4f}")
    print(f"Test Precision (default): {test_precision:.4f}")
    print(f"Test Recall (default): {test_recall:.4f}")
    print(f"Test Accuracy (custom threshold 0.4): {custom_accuracy:.4f}")
    print(f"Test Precision (custom threshold 0.4): {custom_precision:.4f}")
    print(f"Test Recall (custom threshold 0.4): {custom_recall:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('transformer_training_history.png')
    plt.show()
    
    # Save model
    model.save('transformer_ohlcv_model.h5')
    print("\nModel saved as 'transformer_ohlcv_model.h5'")
    
    # Make some predictions with analysis
    print("\nSample predictions:")
    sample_predictions = model.predict(X_test[:10], verbose=0)
    for i, (pred, actual) in enumerate(zip(sample_predictions, y_test[:10])):
        pred_class = 1 if pred[0] > 0.4 else 0
        print(f"Sample {i+1}: Predicted probability = {pred[0]:.4f}, Predicted class = {pred_class}, Actual = {actual}")
    
    # Show prediction distribution
    print(f"\nPrediction distribution:")
    print(f"Min probability: {np.min(sample_predictions):.4f}")
    print(f"Max probability: {np.max(sample_predictions):.4f}")
    print(f"Mean probability: {np.mean(sample_predictions):.4f}")
if __name__ == "__main__":
    train_transformer()