"""
Location Selector Demo - Choose from predefined areas worldwide

This script allows you to select from predefined locations including
Amsterdam, New York, London, Paris, Berlin, and more.
"""

import sys
from config import AREAS, RESOLUTION_NOTES


def list_available_locations():
    """Display all available locations grouped by region"""
    print("\n" + "=" * 80)
    print("AVAILABLE LOCATIONS FOR GREENSPACE ANALYSIS")
    print("=" * 80)
    print()

    # Group locations by region
    regions = {
        'Japan - Yokohama': [],
        'Netherlands - Amsterdam': [],
        'USA - New York City': [],
        'UK - London': [],
        'Germany - Berlin': [],
        'France - Paris': [],
        'Singapore': [],
        'Australia - Sydney': []
    }

    for name, bbox in AREAS.items():
        if bbox is None:
            continue

        if name.startswith('yokohama'):
            regions['Japan - Yokohama'].append((name, bbox))
        elif name.startswith('amsterdam'):
            regions['Netherlands - Amsterdam'].append((name, bbox))
        elif name.startswith('nyc'):
            regions['USA - New York City'].append((name, bbox))
        elif name.startswith('london'):
            regions['UK - London'].append((name, bbox))
        elif name.startswith('berlin'):
            regions['Germany - Berlin'].append((name, bbox))
        elif name.startswith('paris'):
            regions['France - Paris'].append((name, bbox))
        elif name.startswith('singapore'):
            regions['Singapore'].append((name, bbox))
        elif name.startswith('sydney'):
            regions['Australia - Sydney'].append((name, bbox))

    idx = 1
    location_map = {}

    for region, locations in regions.items():
        if locations:
            print(f"\n{region}:")
            print("-" * 80)
            for name, bbox in locations:
                # Calculate approximate area size
                from utils import CoordinateConverter
                area_km2 = CoordinateConverter.calculate_area_km2(bbox)

                # Format name for display
                display_name = name.replace('_', ' ').title()

                print(f"  [{idx:2d}] {display_name:35s} | Area: ~{area_km2:.2f} km²")
                print(f"       Bbox: ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})")

                location_map[idx] = (name, bbox)
                idx += 1

    print("\n" + "=" * 80)
    print()

    # Resolution notes
    print("RESOLUTION INFORMATION:")
    print("-" * 80)
    print(f"Japan (GSI Seamless Photo): {RESOLUTION_NOTES['gsi_seamless']['zoom_18']}")
    print(f"  {RESOLUTION_NOTES['gsi_seamless']['note']}")
    print()
    print("Other regions (require alternative tile sources):")
    for source, info in RESOLUTION_NOTES['alternative_sources'].items():
        print(f"  • {source.replace('_', ' ').title()}: {info}")
    print()
    print("Note: This demo currently uses GSI tiles (Japan only).")
    print("For other regions, modify the tile download function to use alternative sources.")
    print("=" * 80)
    print()

    return location_map


def select_location():
    """Interactive location selection"""
    location_map = list_available_locations()

    while True:
        try:
            choice = input("Select a location number (or 'q' to quit): ").strip()

            if choice.lower() == 'q':
                print("Exiting...")
                sys.exit(0)

            choice_num = int(choice)

            if choice_num in location_map:
                name, bbox = location_map[choice_num]
                return name, bbox
            else:
                print(f"Invalid choice. Please select a number between 1 and {len(location_map)}")

        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def create_custom_location():
    """Create a custom bounding box"""
    print("\n" + "=" * 80)
    print("CUSTOM LOCATION")
    print("=" * 80)
    print()
    print("Enter bounding box coordinates in WGS84 format:")
    print("Format: West (longitude), South (latitude), East (longitude), North (latitude)")
    print()
    print("Example for Amsterdam center: 4.88, 52.36, 4.90, 52.38")
    print()

    while True:
        try:
            bbox_str = input("Enter bbox (west, south, east, north): ").strip()

            if bbox_str.lower() == 'q':
                return None

            coords = [float(x.strip()) for x in bbox_str.split(',')]

            if len(coords) != 4:
                print("Error: Please enter exactly 4 coordinates")
                continue

            west, south, east, north = coords

            # Validate
            if west >= east:
                print("Error: West longitude must be less than East longitude")
                continue
            if south >= north:
                print("Error: South latitude must be less than North latitude")
                continue
            if not (-180 <= west <= 180 and -180 <= east <= 180):
                print("Error: Longitude must be between -180 and 180")
                continue
            if not (-90 <= south <= 90 and -90 <= north <= 90):
                print("Error: Latitude must be between -90 and 90")
                continue

            bbox = (west, south, east, north)

            # Show area size
            from utils import CoordinateConverter
            area_km2 = CoordinateConverter.calculate_area_km2(bbox)
            print(f"\nCustom area size: ~{area_km2:.2f} km²")

            return ('custom', bbox)

        except ValueError:
            print("Error: Invalid format. Please use: west, south, east, north (comma-separated)")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def main():
    """Main interactive demo"""
    print("\n" + "=" * 80)
    print("GREENSPACE ANALYSIS - LOCATION SELECTOR")
    print("=" * 80)
    print()
    print("This tool helps you select a location for greenspace analysis.")
    print()

    while True:
        print("\nOptions:")
        print("  [1] Select from predefined locations")
        print("  [2] Create custom location")
        print("  [q] Quit")
        print()

        choice = input("Your choice: ").strip().lower()

        if choice == '1':
            name, bbox = select_location()
            print(f"\n✓ Selected: {name}")
            print(f"  Bbox: {bbox}")
            print()
            print("To use this location in your analysis:")
            print(f"  area_name = '{name}'")
            print(f"  bbox = {bbox}")
            print()
            print("Or in Python:")
            print(f"  from config import AREAS")
            print(f"  bbox = AREAS['{name}']")
            print()

            run_now = input("Run analysis now? (y/n): ").strip().lower()
            if run_now == 'y':
                run_analysis(name, bbox)
            break

        elif choice == '2':
            result = create_custom_location()
            if result:
                name, bbox = result
                print(f"\n✓ Custom location created")
                print(f"  Bbox: {bbox}")
                print()

                run_now = input("Run analysis now? (y/n): ").strip().lower()
                if run_now == 'y':
                    run_analysis(name, bbox)
            break

        elif choice == 'q':
            print("Exiting...")
            sys.exit(0)

        else:
            print("Invalid choice. Please enter 1, 2, or q")


def run_analysis(area_name, bbox):
    """Run greenspace analysis for selected location"""
    print("\n" + "=" * 80)
    print(f"RUNNING ANALYSIS: {area_name}")
    print("=" * 80)
    print()

    # Check if it's a Japan location (GSI tiles available)
    is_japan = area_name.startswith('yokohama') or 'japan' in area_name.lower()

    if not is_japan:
        print("⚠ WARNING: This location is outside Japan.")
        print("GSI Seamless Photo tiles are only available for Japan.")
        print()
        print("To analyze this location, you need to:")
        print("  1. Modify the tile download function in greenspace_simple.py or greenspace_extraction.py")
        print("  2. Use an alternative tile source (Google, Bing, Mapbox, Sentinel-2)")
        print("  3. Obtain API keys for the chosen service")
        print()
        print("See RESOLUTION_NOTES in config.py for available options.")
        print()

        proceed = input("Continue anyway (for demo/testing)? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Analysis cancelled.")
            return

    print("Importing modules...")
    try:
        from greenspace_simple import get_tiles_for_bbox, train_model, predict_greenspace
        from greenspace_statistics import GreenspaceStatistics, export_metrics_to_csv
        import torch

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        print()

        # Download tiles
        print("Step 1: Downloading tiles...")
        tiles = get_tiles_for_bbox(bbox, zoom=18)

        if not tiles:
            print("ERROR: No tiles downloaded.")
            if not is_japan:
                print("This is expected for non-Japan locations with GSI tiles.")
            return

        print(f"Downloaded {len(tiles)} tiles")
        print()

        # Train model
        print("Step 2: Training model...")
        model = train_model(tiles, epochs=5, device=device)
        print()

        # Generate predictions
        print("Step 3: Generating predictions...")
        predictions = []
        for tile in tiles:
            pred = predict_greenspace(model, tile, device)
            predictions.append(pred)
        print()

        # Calculate statistics
        print("Step 4: Calculating statistics...")
        stats = GreenspaceStatistics(pixel_resolution_m=10.0)
        metrics = stats.calculate_all_metrics(
            predictions=predictions,
            bbox=bbox,
            metadata={'area_name': area_name, 'device': device}
        )

        # Export results
        csv_path = f"output/{area_name}_statistics.csv"
        export_metrics_to_csv(metrics, csv_path)

        print()
        print("=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"Results saved to: {csv_path}")
        print()

    except ImportError as e:
        print(f"ERROR: Failed to import required modules: {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
