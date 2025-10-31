"""
Quick test script to demonstrate the numbered location selector
"""

from config import get_selected_location, SELECTED_LOCATION, LOCATION_LIST

print("=" * 80)
print("LOCATION SELECTOR TEST")
print("=" * 80)
print()

# Show all available locations
print("Available locations:")
print("-" * 80)
for i, (name, bbox) in enumerate(LOCATION_LIST, 1):
    print(f"  [{i:2d}] {name:30s}")
print()

# Show currently selected location
print(f"SELECTED_LOCATION in config.py: {SELECTED_LOCATION}")
print()

# Get the selected location
area_name, bbox = get_selected_location()

print("Current selection:")
print("-" * 80)
print(f"  Area Name: {area_name}")
print(f"  Bounding Box: {bbox}")
print(f"    West:  {bbox[0]}")
print(f"    South: {bbox[1]}")
print(f"    East:  {bbox[2]}")
print(f"    North: {bbox[3]}")
print()

# Show how to change
print("To switch locations:")
print("-" * 80)
print("1. Open config.py")
print("2. Change line 17: SELECTED_LOCATION = X")
print("   Where X is a number from 1 to 19")
print()
print("Examples:")
print("  SELECTED_LOCATION = 1   # Yokohama Station")
print("  SELECTED_LOCATION = 5   # Amsterdam Vondelpark")
print("  SELECTED_LOCATION = 10  # NYC Central Park South")
print("  SELECTED_LOCATION = 13  # London Hyde Park")
print()
print("=" * 80)
