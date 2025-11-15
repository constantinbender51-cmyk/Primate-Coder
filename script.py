import random
import string

def generate_random_matrix():
    # Total characters needed: 50 rows * 100 columns = 5000
    total_chars = 5000
    em_dash_count = 1000
    
    # Generate random characters (excluding em dash for now)
    chars = []
    
    # First, add 4000 random characters from printable ASCII (excluding em dash)
    for _ in range(total_chars - em_dash_count):
        # Use printable characters but avoid em dash
        char = random.choice(string.printable.replace('—', '').replace('\n', '').replace('\r', '').replace('\t', ''))
        chars.append(char)
    
    # Add 1000 em dashes
    for _ in range(em_dash_count):
        chars.append('—')
    
    # Shuffle all characters
    random.shuffle(chars)
    
    # Create 50x100 matrix
    matrix = []
    for i in range(50):
        row = chars[i*100:(i+1)*100]
        matrix.append(row)
    
    return matrix

def display_matrix(matrix):
    print("Random Character Matrix (50x100)")
    print("=" * 100)
    print(f"Total characters: 5000")
    print(f"Hidden em dashes: 1000")
    print("=" * 100)
    print()
    
    for i, row in enumerate(matrix):
        # Join the row characters into a string
        row_str = ''.join(row)
        print(f"Row {i+1:2d}: {row_str}")
    
    print()
    print("=" * 100)
    print("Matrix display complete!")
    
    # Count actual em dashes for verification
    em_dash_total = sum(row.count('—') for row in matrix)
    print(f"Actual em dashes in matrix: {em_dash_total}")

def main():
    print("Generating 5000 random characters with 1000 hidden em dashes...")
    print("Displaying in 50x100 matrix format...")
    print()
    
    matrix = generate_random_matrix()
    display_matrix(matrix)

if __name__ == "__main__":
    main()