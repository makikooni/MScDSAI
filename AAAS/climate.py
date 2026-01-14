import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

URL = "https://en.wikipedia.org/wiki/List_of_cities_by_K%C3%B6ppen_climate_classification"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

print("Downloading Wikipedia page...")
response = requests.get(URL, headers=headers, timeout=20)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

# Köppen climate codes we care about
koppen_codes = [
    "Af", "Am", "Aw",
    "BWh", "BWk", "BSh", "BSk",
    "Csa", "Csb", "Csc",
    "Cfa", "Cfb", "Cfc",
    "Dfa", "Dfb", "Dfc", "Dfd",
    "ET", "EF"
]

all_cities = []
print("Extracting cities from list items...")

for li in soup.find_all("li"):
    links = li.find_all("a")
    if len(links) >= 1:
        city = links[0].get_text(strip=True)
        
        # Skip if city is empty or too short
        if len(city) < 2:
            continue
            
        # Get text after city
        text_after = li.text.replace(city, "").strip()
        
        # Clean up text_after - remove references
        text_after = re.sub(r'\[\d+\]', '', text_after)
        
        # Remove "bordering on" notes
        if "bordering on" in text_after.lower():
            # Try to keep only text before "bordering"
            parts = re.split(r'\(?bordering on', text_after, flags=re.IGNORECASE)
            if parts:
                text_after = parts[0].strip()
        
        # Remove parentheses and their content
        text_after = re.sub(r'\([^)]*\)', '', text_after)
        
        # Clean up commas and spaces
        text_after = text_after.strip()
        if text_after.startswith(','):
            text_after = text_after[1:].strip()
        
        # Extract country - take first part before comma or end
        country = ""
        if text_after:
            # Split by comma and take first meaningful part
            parts = text_after.split(',')
            for part in parts:
                part = part.strip()
                if part and not part.startswith('(') and len(part) > 1:
                    country = part
                    break
        
        # If no country found in text_after, check if there are more links
        if not country and len(links) > 1:
            country = links[1].get_text(strip=True)
        
        # Try to find climate code from parent heading
        heading = li.find_previous("h2")
        if heading:
            heading_text = heading.get_text(strip=True)
            code = None
            for k in koppen_codes:
                if heading_text.startswith(k) or f"{k}:" in heading_text:
                    code = k
                    break
            
            if code and city and country:
                # Additional cleanup
                city = city.strip()
                country = country.strip()
                
                # Remove any trailing punctuation
                city = re.sub(r'[,\s]+$', '', city)
                country = re.sub(r'[,\s]+$', '', country)
                
                # Skip if it's clearly not a city entry
                if (len(city) > 1 and len(country) > 1 and 
                    not city.startswith("List of") and
                    not city.lower().startswith("see also") and
                    "climate" not in city.lower()):
                    
                    all_cities.append({
                        "City": city,
                        "Country": country,
                        "koppen_climate": code
                    })

# Create DataFrame
df = pd.DataFrame(all_cities)

# Remove duplicates (same city-country-climate combination)
df = df.drop_duplicates(subset=["City", "Country", "koppen_climate"])

# Sort the data
df = df.sort_values(["koppen_climate", "City", "Country"])

print(f"\nFound {len(df)} unique city entries")

# Save to CSV
output_file = "cities_by_koppen_climate.csv"
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"Saved to {output_file}")

# Show summary
print("\nClimate code distribution:")
print(df["koppen_climate"].value_counts().sort_index())

print("\nFirst 10 entries:")
print(df.head(10).to_string(index=False))