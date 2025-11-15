import random
import string

def generate_matrix():
    total_chars = 5000
    hidden_em_dashes = 1000
    rows = 10
    cols = 500
    
    # Generate random characters
    all_chars = []
    
    # Add regular random characters
    for _ in range(total_chars - hidden_em_dashes):
        all_chars.append(random.choice(string.printable[:-5]))  # Exclude some special chars
    
    # Add hidden em dashes
    for _ in range(hidden_em_dashes):
        all_chars.append('—')  # Em dash character
    
    # Shuffle all characters
    random.shuffle(all_chars)
    
    # Display header
    print("=" * 100)
    print("Random Character Matrix (10x500)")
    print("=" * 100)
    print(f"Total characters: {total_chars}")
    print(f"Hidden em dashes: {hidden_em_dashes}")
    print("=" * 100)
    
    # Display matrix
    char_index = 0
    em_dash_count = 0
    
    for row in range(rows):
        row_chars = []
        for col in range(cols):
            if char_index < len(all_chars):
                char = all_chars[char_index]
                row_chars.append(char)
                if char == '—':
                    em_dash_count += 1
                char_index += 1
        
        # Display row with row number
        row_display = f"Row {row+1:2d}: {''.join(row_chars)}"
        print(row_display)
    
    print("=" * 100)
    print(f"Actual em dashes in matrix: {em_dash_count}")
    print("=" * 100)

if __name__ == "__main__":
    generate_matrix()