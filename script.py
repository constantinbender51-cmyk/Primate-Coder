import math

def display_numbers_in_circle(numbers, radius=5):
    """
    Display numbers in a circular pattern using ASCII art
    """
    # Calculate positions for numbers in a circle
    positions = []
    n = len(numbers)
    
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        positions.append((x, y, numbers[i]))
    
    # Create a grid to display the numbers
    grid_size = 2 * radius + 1
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Place numbers in the grid
    center = radius
    for x, y, num in positions:
        grid_x = int(round(center + x))
        grid_y = int(round(center + y))
        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
            grid[grid_y][grid_x] = str(num)
    
    # Print the grid
    print("Numbers displayed in a circular pattern:")
    print("-" * 20)
    for row in grid:
        print(' '.join(row))
    print("-" * 20)

# Main execution
if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    display_numbers_in_circle(numbers)
    
    # Additional visualization with more detail
    print("\nDetailed positions:")
    n = len(numbers)
    for i in range(n):
        angle = 360 * i / n
        print(f"Number {numbers[i]} at {angle:.1f}°")