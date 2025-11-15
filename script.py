import random
import string

def generate_matrix():
    # Total characters needed: 25 * 200 = 5000
    total_chars = 5000
    hidden_em_dashes = 1000
    
    # Generate random characters
    all_chars = []
    
    # Add 1000 em dashes
    for _ in range(hidden_em_dashes):
        all_chars.append('—')  # em dash
    
    # Add remaining 4000 random characters
    remaining_chars = total_chars - hidden_em_dashes
    for _ in range(remaining_chars):
        # Include various character types for diversity
        char_type = random.choice(['digit', 'letter', 'symbol'])
        if char_type == 'digit':
            all_chars.append(random.choice(string.digits))
        elif char_type == 'letter':
            all_chars.append(random.choice(string.ascii_letters))
        else:
            all_chars.append(random.choice('!@#$%^&*()_+-=[]{}|;:,.<>?/~`'))
    
    # Shuffle all characters
    random.shuffle(all_chars)
    
    # Display header
    print(f"Generating {total_chars} random characters with {hidden_em_dashes} hidden em dashes...")
    print(f"Displaying in 25x200 matrix format...")
    print("\n" + "=" * 100)
    print("Random Character Matrix (25x200)")
    print("=" * 100)
    print(f"Total characters: {total_chars}")
    print(f"Hidden em dashes: {hidden_em_dashes}")
    print("=" * 100)
    
    # Display matrix
    char_index = 0
    actual_em_dashes = 0
    
    for row in range(25):
        row_chars = all_chars[char_index:char_index + 200]
        char_index += 200
        
        # Count actual em dashes in this row
        row_em_dashes = row_chars.count('—')
        actual_em_dashes += row_em_dashes
        
        # Format row display
        row_display = ''.join(row_chars)
        print(f"Row {row+1:2d}: {row_display}")
    
    print("=" * 100)
    print(f"Matrix display complete!")
    print(f"Actual em dashes in matrix: {actual_em_dashes}")

if __name__ == "__main__":
    generate_matrix()