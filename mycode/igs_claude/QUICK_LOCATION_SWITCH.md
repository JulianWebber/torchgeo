# Quick Location Switching Guide

## How to Switch Locations in One Step

Just change **ONE NUMBER** in `config.py`!

### Step 1: Open config.py

Open the file: `mycode/igs_claude/config.py`

### Step 2: Find Line 17

Look for this line near the top:
```python
SELECTED_LOCATION = 1  # <-- CHANGE THIS NUMBER (1-19) to select location
```

### Step 3: Change the Number

Change the `1` to any number from **1-19**:

```python
SELECTED_LOCATION = 5  # Amsterdam Vondelpark
```

### Step 4: Run Your Script

That's it! Now when you run `run_with_statistics.py` or any script using `get_selected_location()`, it will automatically use your selected location.

## Location Number Reference

### Japan (Works with GSI tiles)
```
1  - yokohama_station
2  - yokohama_park
3  - minato_mirai
4  - yamashita_park
```

### Amsterdam (Requires satellite API)
```
5  - amsterdam_vondelpark          ← Popular choice!
6  - amsterdam_center
7  - amsterdam_westerpark
8  - amsterdam_oost
9  - amsterdam_jordaan
```

### New York City (Requires satellite API)
```
10 - nyc_central_park_south
11 - nyc_central_park_north
12 - nyc_brooklyn_prospect
```

### London (Requires satellite API)
```
13 - london_hyde_park
14 - london_regents_park
15 - london_greenwich_park
```

### Berlin (Requires satellite API)
```
16 - berlin_tiergarten
17 - berlin_tempelhofer_feld
```

### Paris (Requires satellite API)
```
18 - paris_luxembourg_gardens
19 - paris_tuileries
```

## Quick Examples

### Example 1: Switch to Amsterdam Vondelpark
```python
# In config.py, change:
SELECTED_LOCATION = 5
```

### Example 2: Switch to London Hyde Park
```python
# In config.py, change:
SELECTED_LOCATION = 13
```

### Example 3: Switch back to Yokohama
```python
# In config.py, change:
SELECTED_LOCATION = 1
```

## Testing Your Selection

Run this to verify your selection:
```bash
python test_selector.py
```

It will show:
- All 19 available locations
- Your current selection
- The bounding box coordinates

## Using in Your Code

### Automatic (Recommended)
The location is selected automatically when you run:
```bash
python run_with_statistics.py
```

It reads `SELECTED_LOCATION` from config.py automatically!

### Manual (Advanced)
If you want to use it in your own scripts:
```python
from config import get_selected_location

# Automatically gets location based on SELECTED_LOCATION
area_name, bbox = get_selected_location()

print(f"Analyzing: {area_name}")
print(f"Bbox: {bbox}")
```

## Comparison: Old vs New Method

### ❌ Old Method (Hard to remember)
```python
# Had to remember exact text name
area_name = 'amsterdam_vondelpark'
bbox = AREAS[area_name]
```

### ✅ New Method (Just change a number!)
```python
# In config.py:
SELECTED_LOCATION = 5  # Amsterdam Vondelpark

# In your script - nothing to change!
area_name, bbox = get_selected_location()
```

## Tips

1. **Keep it simple**: Just change one number in config.py
2. **Check before running**: Use `python test_selector.py` to verify
3. **Japan locations work immediately**: Numbers 1-4 work with GSI tiles
4. **Other locations need API**: Numbers 5-19 require satellite imagery API
5. **Comments help**: The config file shows comments like `# 5` next to each location

## Full Workflow

1. Open `config.py`
2. Change `SELECTED_LOCATION = 5` (for Amsterdam)
3. Save the file
4. Run `python run_with_statistics.py`
5. Your analysis runs for Amsterdam automatically!

That's it - no code changes needed! 🎉
