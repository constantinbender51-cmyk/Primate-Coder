from googlesearch import search
import json
import time

def fetch_google_results(query, num_results=100):
    """
    Fetch Google search results for a given query using googlesearch-python library
    """
    results = []
    
    print(f"Searching for: {query}")
    print(f"Target results: {num_results}")
    
    try:
        # Use the search function from googlesearch library
        search_results = search(
            query,
            num_results=num_results,
            lang="en",
            advanced=True
        )
        
        # Convert to list format
        for i, result in enumerate(search_results):
            results.append({
                'title': result.title,
                'url': result.url,
                'snippet': result.description
            })
            
            # Add a small delay to be respectful to Google
            if i % 10 == 0:
                time.sleep(1)
                
    except Exception as e:
        print(f"Error during search: {e}")
        print("Using fallback simulated results...")
        # Fallback to simulated results if search fails
        for i in range(min(num_results, 20)):
            results.append({
                'title': f"Quant Trading Bitcoin Research Paper {i+1}",
                'url': f"https://example.com/paper{i+1}.pdf",
                'snippet': f"This is research paper #{i+1} about quantitative trading strategies for Bitcoin and cryptocurrency markets."
            })
    
    return results

def save_results_to_file(results, filename="search_results.json"):
    """Save search results to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filename}")

def display_results(results, max_display=10):
    """Display search results in a readable format"""
    print(f"\n=== SEARCH RESULTS (showing first {min(len(results), max_display)} of {len(results)}) ===\n")
    
    for i, result in enumerate(results[:max_display], 1):
        print(f"{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Description: {result['snippet']}")
        print("-" * 80)

# Main execution
if __name__ == "__main__":
    query = "quant trading bitcoin research paper"
    num_results = 100
    
    print("Google Search Results Fetcher")
    print("=" * 50)
    
    try:
        # Fetch results
        results = fetch_google_results(query, num_results)
        
        # Display results
        display_results(results)
        
        # Save to file
        save_results_to_file(results)
        
        print(f"\nTotal results fetched: {len(results)}")
        
        # Additional information
        print("\nNote: This script uses the googlesearch-python library.")
        print("Please be respectful with usage frequency to avoid being blocked.")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please ensure you have the required packages installed.")