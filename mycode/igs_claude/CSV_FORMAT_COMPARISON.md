# CSV Format Comparison

## New Vertical Format (Default)

The CSV is now organized in **vertical format** with one metric per row, making it much easier to read and analyze.

### Format Structure

```csv
Category,Metric,Value,Full_Metric_Name
```

### Example Output

```csv
Category,Metric,Value,Full_Metric_Name
Metadata,Timestamp,2025-10-31T18:23:54.334194,metadata_timestamp
Metadata,Num Tiles,20,metadata_num_tiles
Metadata,Pixel Resolution M,10.0000,metadata_pixel_resolution_m
Coverage Statistics,Mean Coverage Percent,70.0000,coverage_statistics_mean_coverage_percent
Coverage Statistics,Std Coverage Percent,45.8258,coverage_statistics_std_coverage_percent
Patch Metrics,Num Patches,6,patch_metrics_num_patches
Patch Metrics,Total Greenspace Area Km2,91.7504,patch_metrics_total_greenspace_area_km2
Patch Metrics,Mean Patch Size M2,15291733.3333,patch_metrics_mean_patch_size_m2
Landscape Metrics,Landscape Shape Index,3.0087,landscape_metrics_landscape_shape_index
Landscape Metrics,Aggregation Index,28.5714,landscape_metrics_aggregation_index
Accuracy Metrics,Overall Accuracy,88.3714,accuracy_metrics_overall_accuracy
Accuracy Metrics,F1 Score,0.9233,accuracy_metrics_f1_score
Accuracy Metrics,Cohens Kappa,0.6886,accuracy_metrics_cohens_kappa
```

### Advantages

1. **Easy to Read**: Each metric on its own row
2. **Easy to Filter**: Filter by Category to see specific metric groups
3. **Easy to Sort**: Sort by Category, Metric, or Value
4. **Spreadsheet Friendly**: Works great in Excel, Google Sheets, or LibreOffice
5. **Database Friendly**: Easy to import into databases
6. **Human Readable**: Metric names are formatted for readability (Title Case, spaces)
7. **Reference Column**: Full_Metric_Name provides the original programmatic name

### Using in Excel/Google Sheets

1. Open the CSV file
2. Use **Filter** to select specific categories (e.g., only "Accuracy Metrics")
3. Use **Sort** to organize by category or value
4. Create **Pivot Tables** for summary statistics
5. Generate **Charts** from specific metric categories

### Using in Python (Pandas)

```python
import pandas as pd

# Read the vertical CSV
df = pd.read_csv('output/greenspace_statistics.csv')

# Filter by category
coverage_stats = df[df['Category'] == 'Coverage Statistics']
accuracy_stats = df[df['Category'] == 'Accuracy Metrics']

# Get specific metric value
mean_coverage = df[df['Full_Metric_Name'] == 'coverage_statistics_mean_coverage_percent']['Value'].values[0]

# Group by category
by_category = df.groupby('Category')['Value'].describe()

# Create plots
import matplotlib.pyplot as plt
accuracy_stats.plot(x='Metric', y='Value', kind='bar', title='Accuracy Metrics')
```

### Using in R

```r
library(tidyverse)

# Read CSV
df <- read_csv("output/greenspace_statistics.csv")

# Filter specific category
coverage <- df %>% filter(Category == "Coverage Statistics")

# Get specific value
mean_coverage <- df %>%
  filter(Full_Metric_Name == "coverage_statistics_mean_coverage_percent") %>%
  pull(Value)

# Visualize
df %>%
  filter(Category == "Accuracy Metrics") %>%
  ggplot(aes(x = Metric, y = Value)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  coord_flip()
```

## Old Horizontal Format (Optional)

If you prefer the old format with all metrics in one row (useful for comparing multiple analyses side-by-side), you can still use it:

```python
from greenspace_statistics import export_metrics_to_csv

# Use horizontal format
export_metrics_to_csv(metrics, 'output/statistics.csv', format='horizontal')
```

### Horizontal Format Example

```csv
metadata_timestamp,metadata_num_tiles,coverage_statistics_mean_coverage_percent,patch_metrics_num_patches,...
2025-10-31T18:23:54,20,70.0000,6,...
```

### When to Use Horizontal Format

- Comparing multiple analysis runs side-by-side
- Time series analysis with multiple rows
- Statistical software that expects wide format data
- Appending results from multiple analyses

## Comparison Summary

| Feature | Vertical Format | Horizontal Format |
|---------|----------------|-------------------|
| Readability | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐ Poor (too wide) |
| Easy to filter | ⭐⭐⭐⭐⭐ Yes | ⭐⭐ Difficult |
| Spreadsheet friendly | ⭐⭐⭐⭐⭐ Yes | ⭐⭐⭐ Medium |
| Compare runs | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| Database import | ⭐⭐⭐⭐⭐ Easy | ⭐⭐⭐ Medium |
| Human readable | ⭐⭐⭐⭐⭐ Yes | ⭐ No |
| Single analysis | ⭐⭐⭐⭐⭐ Perfect | ⭐⭐ OK |
| Multiple analyses | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Perfect |

## Recommendation

- **Default (Vertical)**: Best for single analysis runs, journal publication, initial exploration
- **Horizontal**: Best for comparing multiple runs, time series, batch processing

## Converting Between Formats

### Vertical to Horizontal (Python/Pandas)

```python
import pandas as pd

df = pd.read_csv('output/statistics.csv')
horizontal = df.pivot(columns='Full_Metric_Name', values='Value')
horizontal.to_csv('output/statistics_horizontal.csv')
```

### Horizontal to Vertical (Python/Pandas)

```python
import pandas as pd

df = pd.read_csv('output/statistics_horizontal.csv')
vertical = df.melt(var_name='Full_Metric_Name', value_name='Value')

# Add category and metric columns
vertical['Category'] = vertical['Full_Metric_Name'].str.split('_').str[0].str.replace('_', ' ').str.title()
vertical['Metric'] = vertical['Full_Metric_Name'].str.replace(vertical['Category'].str.lower().str.replace(' ', '_') + '_', '').str.replace('_', ' ').str.title()

vertical.to_csv('output/statistics_vertical.csv', index=False)
```
