import pandas as pd

# Load all datasets
print("Loading datasets...")

# 1. Countries data (with country codes, lat, long)
countries = pd.read_csv('AAAS/countries.csv')
print(f"Countries data: {countries.shape[0]} rows")

# 2. Cities with Köppen climate data
cities_climate = pd.read_csv('AAAS/cities_by_koppen_climate.csv')
print(f"Cities climate data: {cities_climate.shape[0]} rows")

# 3. Hofstede cultural dimensions
hofstede = pd.read_csv('AAAS/hofstede.csv')
print(f"Hofstede data: {hofstede.shape[0]} rows")

# 4. Income data
income = pd.read_csv('AAAS/income.csv')
print(f"Income data: {income.shape[0]} rows")

# ------------------------------------------------------------
# Clean country names function
# ------------------------------------------------------------
def clean_country_name(name):
    """Clean and standardize country names"""
    name = str(name).strip()
    # Remove any trailing punctuation or notes in parentheses
    name = name.split('(')[0].strip()
    name = name.split('[')[0].strip()
    
    # Common country name mappings
    country_name_mapping = {
        'United States': 'United States of America',
        'USA': 'United States of America',
        'U.S.': 'United States of America',
        'U.K.': 'United Kingdom',
        'UK': 'United Kingdom',
        'South Korea': 'Korea, Republic of',
        'Russia': 'Russian Federation',
        'Czech Republic': 'Czechia',
        'Iran': 'Iran, Islamic Republic of',
        'Vietnam': 'Viet Nam',
        'Macedonia': 'North Macedonia',
        'Laos': "Lao People's Democratic Republic",
        'Syria': 'Syrian Arab Republic',
        'Venezuela': 'Venezuela, Bolivarian Republic of',
        'Tanzania': 'Tanzania, United Republic of',
        'Bolivia': 'Bolivia, Plurinational State of',
        'Brunei': 'Brunei Darussalam',
    }
    
    if name in country_name_mapping:
        return country_name_mapping[name]
    return name

# ------------------------------------------------------------
# STEP 1: Prepare each dataset
# ------------------------------------------------------------

# 1. Clean countries dataset (we need country code, lat, long, and name)
countries_clean = countries.copy()
countries_clean['name_clean'] = countries_clean['name'].apply(clean_country_name)

# 2. Clean cities dataset
cities_clean = cities_climate.copy()
cities_clean['Country_clean'] = cities_clean['Country'].apply(clean_country_name)

# 3. Clean Hofstede dataset
hofstede_clean = hofstede.copy()
hofstede_clean['country_clean'] = hofstede_clean['country'].apply(clean_country_name)

# 4. Clean income dataset
income_clean = income.copy()
income_clean['Country_clean'] = income_clean['Economy'].apply(clean_country_name)

# ------------------------------------------------------------
# STEP 2: Merge step by step
# ------------------------------------------------------------

print("\nMerging datasets...")

# Start with cities data as base
merged = cities_clean.copy()

# Add country code, latitude, and longitude from countries dataset
print("1. Adding country codes and coordinates...")
merged = pd.merge(
    merged,
    countries_clean[['name_clean', 'country', 'latitude', 'longitude']],
    left_on='Country_clean',
    right_on='name_clean',
    how='left'
)

# Clean up column names
merged = merged.rename(columns={
    'country': 'COUNTRY_CODE',
    'Country_clean': 'COUNTRY',
    'City': 'CITY',
    'koppen_climate': 'CLIMATE_CODE',
    'latitude': 'LATITUDE',
    'longitude': 'LONGITUDE'
})

# Select only needed columns (we'll keep all for now, reorder later)
merged = merged[['COUNTRY_CODE', 'COUNTRY', 'CITY', 'LATITUDE', 'LONGITUDE', 'CLIMATE_CODE']]

# Add Hofstede dimensions
print("2. Adding Hofstede dimensions...")
merged = pd.merge(
    merged,
    hofstede_clean[['country_clean', 'pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr']],
    left_on='COUNTRY',
    right_on='country_clean',
    how='left'
)

# Remove the temporary column
merged = merged.drop(columns=['country_clean'])

# Add income group
print("3. Adding income group...")
merged = pd.merge(
    merged,
    income_clean[['Country_clean', 'Income group']],
    left_on='COUNTRY',
    right_on='Country_clean',
    how='left'
)

# Rename income column and drop temporary column
merged = merged.rename(columns={'Income group': 'INCOME_GROUP'})
merged = merged.drop(columns=['Country_clean'])

# ------------------------------------------------------------
# STEP 3: Remove rows with missing data
# ------------------------------------------------------------

print(f"\nOriginal dataset size: {merged.shape[0]} rows")

# Check missing data before removal
print("\nMissing data BEFORE removal:")
print(f"  COUNTRY_CODE missing: {merged['COUNTRY_CODE'].isna().sum()}")
print(f"  LATITUDE missing: {merged['LATITUDE'].isna().sum()}")
print(f"  LONGITUDE missing: {merged['LONGITUDE'].isna().sum()}")
print(f"  INCOME_GROUP missing: {merged['INCOME_GROUP'].isna().sum()}")
print(f"  Hofstede data missing: {merged['pdi'].isna().sum()}")

# Remove rows with ANY missing data
merged_clean = merged.dropna()

print(f"\nAfter removing rows with missing data: {merged_clean.shape[0]} rows")
print(f"Rows removed: {merged.shape[0] - merged_clean.shape[0]}")

# ------------------------------------------------------------
# STEP 4: Final cleanup and ordering
# ------------------------------------------------------------

# Reorder columns as requested: COUNTRY_CODE, COUNTRY, CITY, LATITUDE, LONGITUDE, INCOME_GROUP, CLIMATE_CODE, Hofstede dimensions
final_columns = [
    'COUNTRY_CODE', 'COUNTRY', 'CITY', 'LATITUDE', 'LONGITUDE', 
    'INCOME_GROUP', 'CLIMATE_CODE', 
    'pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr'
]

# Create final dataframe with proper column order
final_dataset = merged_clean[final_columns]

# Sort for readability
final_dataset = final_dataset.sort_values(['COUNTRY', 'CITY', 'CLIMATE_CODE'])

# Reset index for clean output
final_dataset = final_dataset.reset_index(drop=True)

# ------------------------------------------------------------
# STEP 5: Save and show preview
# ------------------------------------------------------------

output_file = "country_city_climate_culture_income_coords_COMPLETE.csv"
final_dataset.to_csv(output_file, index=False, encoding='utf-8')

print(f"\nDataset created successfully!")
print(f"Saved to: {output_file}")
print(f"\nDataset shape: {final_dataset.shape[0]} rows, {final_dataset.shape[1]} columns")
print(f"Unique countries: {final_dataset['COUNTRY'].nunique()}")
print(f"Unique cities: {final_dataset['CITY'].nunique()}")
print(f"Climate codes: {final_dataset['CLIMATE_CODE'].nunique()}")

print("\n" + "="*70)
print("FIRST 15 ROWS OF THE COMPLETE DATASET (NO MISSING DATA):")
print("="*70)
print(final_dataset.head(15).to_string(index=False))

print("\n" + "="*70)
print("COLUMNS IN FINAL DATASET:")
print("="*70)
for i, col in enumerate(final_dataset.columns, 1):
    print(f"{i:2}. {col}")

# Show summary statistics of complete dataset
print("\n" + "="*70)
print("SUMMARY OF COMPLETE DATASET:")
print("="*70)
print(f"Total complete records: {final_dataset.shape[0]}")
print(f"Countries represented: {final_dataset['COUNTRY'].nunique()}")
print(f"Cities represented: {final_dataset['CITY'].nunique()}")

# Show climate distribution
print("\nClimate code distribution in complete dataset:")
climate_counts = final_dataset['CLIMATE_CODE'].value_counts().sort_index()
for climate, count in climate_counts.items():
    print(f"  {climate}: {count} cities")

# Show income group distribution
print("\nIncome group distribution in complete dataset:")
income_counts = final_dataset['INCOME_GROUP'].value_counts()
for income_group, count in income_counts.items():
    print(f"  {income_group}: {count} cities")

# Verify no missing data
print("\n" + "="*70)
print("VERIFICATION - NO MISSING DATA IN FINAL DATASET:")
print("="*70)
missing_check = final_dataset.isna().sum()
if missing_check.sum() == 0:
    print("✓ All data is complete - no missing values!")
else:
    print("✗ Some missing data found:")
    for col, missing in missing_check.items():
        if missing > 0:
            print(f"  {col}: {missing} missing")