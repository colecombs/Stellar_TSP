import csv
import math
import time
import pathlib
from typing import Any, Dict, Optional
import argparse

# --- Physical Constants ---
# Using standard values for high precision.
C_METERS_PER_SECOND = 299792458.0
G_ACCELERATION = 9.80665  # Standard gravity in m/s^2
SECONDS_PER_YEAR = 31557600.0  # 365.25 days
METERS_PER_LIGHT_YEAR = C_METERS_PER_SECOND * SECONDS_PER_YEAR
KM_PER_LY = METERS_PER_LIGHT_YEAR / 1000
ARCSECONDS_PER_DEGREE = 3600.0

def hms_to_decimal(hms_str: str) -> Optional[float]:
    """
    Converts a Right Ascension string (hh:mm:ss.s) to decimal degrees.
    Returns None if the format is invalid.
    """
    # Handle the special case for Sol or other zero-RA objects
    if hms_str.strip() == '0':
        return 0.0

    try:
        hours, minutes, seconds = map(float, hms_str.split(':'))
        if not (0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60):
            print(f"Warning: Invalid HMS value found: {hms_str}")
            return None
        return 15 * (hours + minutes / 60 + seconds / 3600)
    except (ValueError, TypeError):
        print(f"Warning: Could not parse HMS string: '{hms_str}'")
        return None

def load_star_data(filename: str = "closest_stars.csv") -> Dict[str, Dict[str, Any]]:
    """
    Loads star data from a CSV and converts spherical coordinates to Cartesian.

    The CSV is expected to have 'star_name', 'distance_ly', 'ra_hms', 'dec_deg',
    'RA_Motion', 'Dec_Motion', and 'Rad_v(km/s)'.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        A dictionary mapping star names to their data. Each star's data is a
        dictionary containing:
        - 'coords': A tuple (x, y, z) in lightyears.
        - 'distance_ly': Original distance from Sol in lightyears (at t=0).
        - 'ra_deg': Right Ascension in decimal degrees.
        - 'dec_deg': Declination in decimal degrees.
        - 'ra_motion_mas_yr': Proper motion in RA (milliarcseconds/year).
        - 'dec_motion_mas_yr': Proper motion in Dec (milliarcseconds/year).
        - 'rad_v_km_s': Radial velocity (km/s).
    """
    stars = {}
    # Use 'utf-8-sig' to automatically handle the Byte Order Mark (BOM)
    with open(filename, mode='r', encoding='utf-8-sig', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            try:
                star_name = row['star_name']
                dist_ly = float(row['distance_ly'])
                ra_deg = hms_to_decimal(row['ra_hms'])
                dec_deg = float(row['dec_deg'])
                # Use .get() to provide a default of '0' if motion data is missing
                ra_motion = float(row.get('RA_Motion', '0'))
                dec_motion = float(row.get('Dec_Motion', '0'))
                rad_v_km_s = float(row.get('Rad_v(km/s)', '0'))

                # Add a check to ensure conversion was successful before proceeding
                if ra_deg is None:
                    continue # Skip to the next row

                # Convert spherical to Cartesian coordinates
                # First, convert degrees to radians
                ra_rad = math.radians(ra_deg)
                dec_rad = math.radians(dec_deg)

                # Now, calculate x, y, z
                x = dist_ly * math.cos(dec_rad) * math.cos(ra_rad)
                y = dist_ly * math.cos(dec_rad) * math.sin(ra_rad)
                z = dist_ly * math.sin(dec_rad)

                stars[star_name] = {
                    'coords': (x, y, z),
                    'distance_ly': dist_ly,
                    'ra_deg': ra_deg,
                    'dec_deg': dec_deg,
                    'ra_motion_mas_yr': ra_motion,
                    'dec_motion_mas_yr': dec_motion,
                    'rad_v_km_s': rad_v_km_s,
                }
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {row} - {e}")
    return stars

def predict_star_position(star_data: Dict[str, Any], time_years: float) -> tuple[float, float, float]:
    """
    Predicts a star's future Cartesian coordinates based on its proper motion
    and radial velocity, as observed from the Sol reference frame.

    Args:
        star_data (Dict): The dictionary for a single star from load_star_data.
        time_years (float): The number of years into the future from t=0.

    Returns:
        A tuple (x, y, z) of the star's predicted coordinates in lightyears.
    """
    # If time is zero, or for Sol, return initial coordinates
    if time_years == 0 or star_data.get('distance_ly') == 0:
        return star_data['coords']

    # 1. Calculate change in distance due to radial velocity
    rad_v_ly_per_year = star_data['rad_v_km_s'] * (SECONDS_PER_YEAR / KM_PER_LY)
    new_dist_ly = star_data['distance_ly'] + (rad_v_ly_per_year * time_years)

    # 2. Calculate change in RA and Dec due to proper motion
    initial_ra_deg = star_data['ra_deg']
    initial_dec_deg = star_data['dec_deg']

    # Convert motion from milliarcseconds/year to degrees/year
    ra_motion_deg_per_year = star_data['ra_motion_mas_yr'] / (1000 * ARCSECONDS_PER_DEGREE)
    dec_motion_deg_per_year = star_data['dec_motion_mas_yr'] / (1000 * ARCSECONDS_PER_DEGREE)

    # Update RA and Dec coordinates
    delta_ra_deg = ra_motion_deg_per_year * time_years
    delta_dec_deg = dec_motion_deg_per_year * time_years

    new_ra_deg = initial_ra_deg + delta_ra_deg
    new_dec_deg = initial_dec_deg + delta_dec_deg

    # 3. Convert new spherical coordinates back to Cartesian
    new_ra_rad = math.radians(new_ra_deg)
    new_dec_rad = math.radians(new_dec_deg)

    x = new_dist_ly * math.cos(new_dec_rad) * math.cos(new_ra_rad)
    y = new_dist_ly * math.cos(new_dec_rad) * math.sin(new_ra_rad)
    z = new_dist_ly * math.sin(new_dec_rad)

    return x, y, z

def calculate_distance(coords1, coords2):
    """Calculates the 3D Euclidean distance between two points."""
    return math.sqrt(
        (coords1[0] - coords2[0])**2 +
        (coords1[1] - coords2[1])**2 +
        (coords1[2] - coords2[2])**2
    )

def solve_tsp_nearest_neighbor(stars, start_star):
    """
    Solves the TSP using the Nearest Neighbor heuristic.

    Args:
        stars (dict): A dictionary of star data from load_star_data.
        start_star (str): The name of the starting star.

    Returns:
        A tuple containing the ordered tour (list of star names) and the
        total distance of the tour in lightyears.
    """
    if start_star not in stars:
        raise ValueError("Start star not found in the dataset.")

    unvisited = set(stars.keys())
    tour = [start_star]
    total_distance = 0.0
    current_star_name = start_star
    unvisited.remove(current_star_name)

    while unvisited:
        nearest_star = None
        min_dist = float('inf')

        current_coords = stars[current_star_name]['coords']
        for star_name in unvisited:
            dist = calculate_distance(current_coords, stars[star_name]['coords'])
            if dist < min_dist:
                min_dist = dist
                nearest_star = star_name

        total_distance += min_dist
        current_star_name = nearest_star
        tour.append(current_star_name)
        unvisited.remove(current_star_name)

    # Add distance from the last star back to the start
    last_star_coords = stars[tour[-1]]['coords']
    start_star_coords = stars[start_star]['coords']
    total_distance += calculate_distance(last_star_coords, start_star_coords)
    tour.append(start_star) # Complete the loop

    return tour, total_distance

def calculate_relativistic_travel_time(distance_ly):
    """
    Calculates ship time (proper time) and Sol time (coordinate time) for a
    trip with constant 1g acceleration for the first half and 1g deceleration
    for the second half.

    Args:
        distance_ly (float): The distance of the travel leg in lightyears.

    Returns:
        A tuple containing (ship_time_years, sol_time_years).
    """
    # Convert distance from lightyears to meters
    d_meters = distance_ly * METERS_PER_LIGHT_YEAR

    # If distance is zero, travel time is zero
    if d_meters == 0:
        return 0.0, 0.0

    # Use shorter variable names for the physics equations
    a = G_ACCELERATION
    c = C_METERS_PER_SECOND

    # This term appears in both the proper and coordinate time equations
    term = (a * d_meters) / (2 * c**2)

    # Proper time (τ) for the ship's crew for one leg, converted to years
    ship_time_leg_sec = 2 * (c / a) * math.acosh(1 + term)
    ship_time_leg_years = ship_time_leg_sec / SECONDS_PER_YEAR

    # Coordinate time (t) for a stationary observer (Sol), converted to years
    sol_time_leg_sec = 2 * (c / a) * math.sqrt((1 + term)**2 - 1)
    sol_time_leg_years = sol_time_leg_sec / SECONDS_PER_YEAR

    return ship_time_leg_years, sol_time_leg_years

def calculate_leg_details(distance_ly):
    """
    Calculates all travel metrics for a single leg of the journey.

    Args:
        distance_ly (float): The distance of the travel leg in lightyears.

    Returns:
        A dictionary containing detailed metrics for the leg.
    """
    ship_time_years, sol_time_years = calculate_relativistic_travel_time(distance_ly)

    # Avoid division by zero for zero-distance legs or if ship_time is zero
    if ship_time_years == 0:
        return {
            'distance_ly': distance_ly,
            'ship_time_years': 0.0,
            'sol_time_years': 0.0,
            'max_speed_percent_c': 0.0,
            'time_dilation_factor': 1.0
        }

    # Calculate time dilation factor (how much faster time passes for Sol)
    time_dilation_factor = sol_time_years / ship_time_years

    # Calculate max speed as a percentage of the speed of light
    # v_max = c * tanh(a * tau_half / c)
    # tau_half is half the proper time (ship time) for the leg in seconds
    ship_time_half_sec = (ship_time_years * SECONDS_PER_YEAR) / 2
    tanh_arg = (G_ACCELERATION * ship_time_half_sec) / C_METERS_PER_SECOND
    max_speed_fraction_c = math.tanh(tanh_arg)
    max_speed_percent_c = max_speed_fraction_c * 100

    return {
        'distance_ly': distance_ly,
        'ship_time_years': ship_time_years,
        'sol_time_years': sol_time_years,
        'max_speed_percent_c': max_speed_percent_c,
        'time_dilation_factor': time_dilation_factor
    }

def calculate_total_tour_time(tour, stars):
    """
    Calculates the total relativistic travel time and rest time for an entire tour.
    """
    total_ship_time_years = 0.0
    total_sol_time_years = 0.0
    cumulative_sol_time = 0.0  # Time elapsed in Sol's frame since tour start
    leg_details = [] # To store detailed metrics for each leg

    # Calculate travel time for each leg of the journey
    for i in range(len(tour) - 1):
        star1_name, star2_name = tour[i], tour[i+1]

        # Position of departure star AT THE TIME OF DEPARTURE
        coords1 = predict_star_position(stars[star1_name], cumulative_sol_time)

        # --- Iterative estimation to find the interception point ---
        # 1. First guess: travel time to star2's current position (at departure time)
        coords2_guess = predict_star_position(stars[star2_name], cumulative_sol_time)
        dist_guess = calculate_distance(coords1, coords2_guess)
        _, sol_time_guess = calculate_relativistic_travel_time(dist_guess)

        # 2. Second, better guess: predict star2's position at the estimated arrival time
        estimated_arrival_time = cumulative_sol_time + sol_time_guess
        coords2_final = predict_star_position(stars[star2_name], estimated_arrival_time)

        # 3. Final calculation for the leg using the interception point
        final_distance = calculate_distance(coords1, coords2_final)
        details = calculate_leg_details(final_distance)
        leg_details.append(details)

        # Update totals and cumulative time with this leg's travel time
        total_ship_time_years += details['ship_time_years']
        total_sol_time_years += details['sol_time_years']
        cumulative_sol_time += details['sol_time_years']

        # Add 1-year rest time if this is an intermediate stop (not the final return to Sol)
        if i < len(tour) - 2:
            rest_time = 1.0
            total_ship_time_years += rest_time
            total_sol_time_years += rest_time
            cumulative_sol_time += rest_time

    return total_ship_time_years, total_sol_time_years, leg_details

def print_tour_summary(tour, distance, leg_details):
    """
    Prints a formatted summary of the calculated TSP tour, including detailed leg data.
    """
    if len(tour) - 1 != len(leg_details):
        print("Warning: Mismatch between tour legs and leg details data.")
        return

    print("\n--- Traveling Salesman Star Tour ---")
    print(f"Total tour distance: {distance:.2f} lightyears")
    print("Tour path: " + " -> ".join(tour))

    # Print a detailed table for each leg
    print("\n--- Leg-by-Leg Analysis ---")
    header = (
        f"{'Leg':<38s} | {'Distance (ly)':>15s} | {'Ship Time (y)':>15s} | "
        f"{'Sol Time (y)':>15s} | {'Max Speed (%c)':>15s} | {'Time Dilation':>15s}"
    )
    print(header)
    print("-" * len(header))

    for i, details in enumerate(leg_details):
        leg_name = f"{tour[i]} -> {tour[i+1]}"
        dist_str = f"{details['distance_ly']:.2f}"
        ship_time_str = f"{details['ship_time_years']:.2f}"
        sol_time_str = f"{details['sol_time_years']:.2f}"
        max_speed_str = f"{details['max_speed_percent_c']:.2f}"
        dilation_str = f"{details['time_dilation_factor']:.2f}"

        row = (
            f"{leg_name:<38s} | {dist_str:>15s} | {ship_time_str:>15s} | "
            f"{sol_time_str:>15s} | {max_speed_str:>15s} | {dilation_str:>15s}"
        )
        print(row)

def print_execution_time(start_time, end_time):
    """
    Calculates and prints the script's total execution time in milliseconds.
    """
    duration_s = end_time - start_time
    duration_ms = duration_s * 1000
    print("\n--- Script Execution Time ---")
    print(f"Total time: {duration_ms:.3f} milliseconds")

if __name__ == "__main__":
    # Start the high-resolution performance counter
    start_time = time.perf_counter()

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Solve the Traveling Salesman Problem for a list of stars from a CSV file."
    )
    parser.add_argument("csv_file", help="Path to the CSV file containing star data.")
    parser.add_argument(
        "--start_star", default="Sol", help="The name of the star to start the tour from (default: Sol)."
    )
    args = parser.parse_args()

    # --- Robust Path Handling ---
    # Create a path object from the user's input
    csv_path = pathlib.Path(args.csv_file)

    # If the provided path is not absolute, assume it's relative to the script's location
    if not csv_path.is_absolute():
        script_dir = pathlib.Path(__file__).parent.resolve()
        csv_path = script_dir / csv_path

    # 1. Load the star data from the CSV
    print(f"Loading star data from '{csv_path}'...")
    all_stars = load_star_data(csv_path)
    print(f"Loaded data for {len(all_stars)} stars.")

    # 2. Define the starting point and solve the TSP
    print(f"\nCalculating shortest tour starting from {args.start_star} using Nearest Neighbor heuristic...")
    try:
        best_tour, total_dist = solve_tsp_nearest_neighbor(all_stars, args.start_star)

        # 3. Calculate relativistic travel times for the tour, including individual legs
        ship_time, sol_time, leg_details = calculate_total_tour_time(best_tour, all_stars)

        # Print the detailed tour summary using the leg details
        print_tour_summary(best_tour, total_dist, leg_details)

        # --- Calculate overall tour averages from leg details ---
        total_travel_ship_time = sum(d['ship_time_years'] for d in leg_details)
        total_travel_sol_time = sum(d['sol_time_years'] for d in leg_details)

        # Avoid division by zero if there's no travel time
        average_dilation = (total_travel_sol_time / total_travel_ship_time) if total_travel_ship_time > 0 else 1.0

        # Weighted average of max speed, weighted by Sol time for each leg
        # This gives a sense of the "typical" max speed over the whole journey duration.
        weighted_avg_max_speed = (sum(d['max_speed_percent_c'] * d['sol_time_years'] for d in leg_details) / total_travel_sol_time) if total_travel_sol_time > 0 else 0.0

        # Print the final analysis using the total times
        print("\n--- Relativistic Travel Time Analysis ---")
        print("Assumptions: 1g constant accel/decel, 1-year stay at each destination.")
        print(f"Total subjective time on ship (travel + stops): {ship_time:.2f} years")
        print(f"Total time in Sol's frame (travel + stops):   {sol_time:.2f} years")
        print(f"Average time dilation factor (travel only):     {average_dilation:.2f}x")
        print(f"Average max speed (weighted by leg duration):   {weighted_avg_max_speed:.2f}% of c")

    except ValueError as e:
        print(f"\nError: {e}")

    # Stop the counter and print the elapsed time
    end_time = time.perf_counter()
    print_execution_time(start_time, end_time)