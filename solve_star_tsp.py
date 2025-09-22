import csv
import math
import time
import pathlib
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import collections
import argparse

# --- Physical Constants ---
# Using standard values for high precision.
C_METERS_PER_SECOND = 299792458.0
G_ACCELERATION = 9.80665  # Standard gravity in m/s^2
SECONDS_PER_YEAR = 31557600.0  # 365.25 days
METERS_PER_LIGHT_YEAR = C_METERS_PER_SECOND * SECONDS_PER_YEAR

# --- Animation Constants ---
TRAIL_LENGTH = 150  # Number of points in the ship's trail
ANIMATION_INTERVAL_MS = 20  # Milliseconds between frames (50 FPS)
SIM_YEARS_PER_SECOND = 10.0 # How many years of sim time pass per second of animation

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

    The CSV is expected to have 'star_name', 'distance_ly', 'ra_hms', 'dec_deg'.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        A dictionary mapping star names to their data. Each star's data is a
        dictionary containing:
        - 'coords': A tuple (x, y, z) in lightyears.
        - 'distance_ly': Original distance from Sol in lightyears.
        - 'ra_deg': Right Ascension in decimal degrees.
        - 'dec_deg': Declination in decimal degrees.
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
                    'dec_deg': dec_deg
                }
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {row} - {e}")
    return stars

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
    leg_details = [] # To store detailed metrics for each leg

    # Add the 1-year rest period for each star visited (excluding start and end at Sol)
    num_stops = len(tour) - 2
    if num_stops > 0:
        # This rest time is the same for both ship and Sol reference frames
        rest_time = float(num_stops)
        total_ship_time_years += rest_time
        total_sol_time_years += rest_time

    # Calculate travel time for each leg of the journey
    for i in range(len(tour) - 1):
        star1_name, star2_name = tour[i], tour[i+1]
        coords1, coords2 = stars[star1_name]['coords'], stars[star2_name]['coords']

        distance_ly = calculate_distance(coords1, coords2)
        details = calculate_leg_details(distance_ly)

        leg_details.append(details)
        total_ship_time_years += details['ship_time_years']
        total_sol_time_years += details['sol_time_years']

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
        f"{'Leg':<28s} | {'Distance (ly)':>15s} | {'Ship Time (y)':>15s} | "
        f"{'Sol Time (y)':>15s} | {'Max Speed (%c)':>15s} | {'Time Dilation':>15s}"
    )
    print(header)
    print("-" * len(header))

    for i, details in enumerate(leg_details):
        leg_name = f"{tour[i]:>12s} -> {tour[i+1]:<12s}"
        dist_str = f"{details['distance_ly']:.2f}"
        ship_time_str = f"{details['ship_time_years']:.2f}"
        sol_time_str = f"{details['sol_time_years']:.2f}"
        max_speed_str = f"{details['max_speed_percent_c']:.2f}"
        dilation_str = f"{details['time_dilation_factor']:.2f}"

        row = (
            f"{leg_name:<28s} | {dist_str:>15s} | {ship_time_str:>15s} | "
            f"{sol_time_str:>15s} | {max_speed_str:>15s} | {dilation_str:>15s}"
        )
        print(row)

def get_ship_position_at_sol_time(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    total_sol_time_leg: float,
    time_into_leg_years: float
) -> np.ndarray:
    """
    Calculates the ship's 3D position at a specific coordinate time (Sol time)
    within a leg, assuming 1g constant proper acceleration/deceleration.

    Args:
        start_pos (np.ndarray): The 3D coordinates of the departure star.
        end_pos (np.ndarray): The 3D coordinates of the arrival star.
        total_sol_time_leg (float): The total Sol time for this leg in years.
        time_into_leg_years (float): The elapsed Sol time since the leg began.

    Returns:
        np.ndarray: The calculated 3D position of the ship in lightyears.
    """
    direction_vector = end_pos - start_pos
    total_distance_ly = np.linalg.norm(direction_vector)

    if total_distance_ly == 0 or total_sol_time_leg == 0:
        return start_pos

    unit_direction = direction_vector / total_distance_ly

    # Acceleration in ly/year^2 for 1g
    a_ly_per_year_sq = G_ACCELERATION * (SECONDS_PER_YEAR**2) / METERS_PER_LIGHT_YEAR
    c_ly_per_year = 1.0  # By definition

    def dist_from_rest(t_years: float) -> float:
        """
        Calculates relativistic distance traveled from rest at coordinate time t.
        Equation: d(t) = (c^2/a) * [sqrt(1 + (a*t/c)^2) - 1]
        """
        if t_years <= 0:
            return 0.0
        # All units are in lightyears and years, so c=1.
        term = (a_ly_per_year_sq * t_years)**2
        dist_ly = (c_ly_per_year**2 / a_ly_per_year_sq) * (np.sqrt(1 + term) - 1)
        return dist_ly

    midpoint_sol_time = total_sol_time_leg / 2.0

    if time_into_leg_years <= midpoint_sol_time:
        # Acceleration phase (first half of the journey)
        distance_traveled = dist_from_rest(time_into_leg_years)
    else:
        # Deceleration phase (second half)
        # We calculate the distance from the destination by considering the time
        # remaining in the leg.
        time_from_end = total_sol_time_leg - time_into_leg_years
        distance_from_end = dist_from_rest(time_from_end)
        distance_traveled = total_distance_ly - distance_from_end

    # Clamp the distance to prevent overshooting due to floating point errors
    distance_traveled = np.clip(distance_traveled, 0, total_distance_ly)

    current_pos = start_pos + distance_traveled * unit_direction
    return current_pos

def run_animation(all_stars, tour, leg_details):
    """Sets up and runs the Matplotlib 3D tour animation."""
    # --- 1. Pre-calculate the full tour schedule for animation ---
    tour_schedule = []
    cumulative_sol_time = 0.0
    for i, leg in enumerate(leg_details):
        start_star_name, end_star_name = tour[i], tour[i+1]
        start_pos = np.array(all_stars[start_star_name]['coords'])
        end_pos = np.array(all_stars[end_star_name]['coords'])
        leg_sol_time = leg['sol_time_years']
        arrival_sol_time = cumulative_sol_time + leg_sol_time

        tour_schedule.append({
            'start_time': cumulative_sol_time,
            'end_time': arrival_sol_time,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'duration': leg_sol_time,
        })

        cumulative_sol_time = arrival_sol_time
        # Add 1-year rest time if not the final leg back to Sol
        if i < len(leg_details) - 1:
            cumulative_sol_time += 1.0
    total_sim_duration_years = cumulative_sol_time

    # --- 2. Set up the Matplotlib 3D plot ---
    fig = plt.figure(figsize=(12, 12), facecolor='black')
    ax = fig.add_subplot(projection='3d', facecolor='black')
    ax.set_title("Relativistic Star Tour", color='white')
    ax.set_xlabel("X (ly)", color='white')
    ax.set_ylabel("Y (ly)", color='white')
    ax.set_zlabel("Z (ly)", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    view_radius_ly = 20.0
    ax.set_xlim(-view_radius_ly, view_radius_ly)
    ax.set_ylim(-view_radius_ly, view_radius_ly)
    ax.set_zlim(-view_radius_ly, view_radius_ly)

    star_coords = np.array([s['coords'] for s in all_stars.values()])
    ax.scatter(star_coords[:, 0], star_coords[:, 1], star_coords[:, 2], s=10, c='white', alpha=0.7)
    for name, star in all_stars.items():
        if name in set(tour):
            ax.text(star['coords'][0], star['coords'][1], star['coords'][2], f' {name}', color='cyan', fontsize=9)

    # --- 3. Prepare for animation ---
    trail_deque = collections.deque(maxlen=TRAIL_LENGTH)
    trail_lines = [ax.plot([], [], [], lw=2)[0] for _ in range(TRAIL_LENGTH - 1)]
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white')

    # --- 4. Define the animation update function ---
    total_frames = int((total_sim_duration_years / SIM_YEARS_PER_SECOND) * (1000 / ANIMATION_INTERVAL_MS))

    def animate(frame_num):
        current_sim_time = (frame_num / total_frames) * total_sim_duration_years
        current_leg = next((leg for leg in tour_schedule if leg['start_time'] <= current_sim_time < leg['end_time']), None)

        if current_leg:
            time_into_leg = current_sim_time - current_leg['start_time']
            ship_pos = get_ship_position_at_sol_time(current_leg['start_pos'], current_leg['end_pos'], current_leg['duration'], time_into_leg)
            is_accel = time_into_leg < current_leg['duration'] / 2.0
            trail_deque.append((ship_pos, is_accel))
        elif trail_deque:
            ship_pos, is_accel = trail_deque[-1]
            trail_deque.append((ship_pos, is_accel))

        for i, line in enumerate(trail_lines):
            if i < len(trail_deque) - 1:
                p1, _ = trail_deque[i]
                p2, is_accel_p2 = trail_deque[i+1]
                line.set_data_3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
                alpha = (i + 1) / TRAIL_LENGTH
                color = 'lime' if is_accel_p2 else 'red'
                line.set_color(color); line.set_alpha(alpha)
            else:
                line.set_data_3d([], [], [])

        time_text.set_text(f'Sol Time: {current_sim_time:.2f} years')
        return trail_lines + [time_text]

    # --- 5. Run the animation ---
    ani = FuncAnimation(fig, animate, frames=total_frames, interval=ANIMATION_INTERVAL_MS, blit=True, repeat=False)
    print("\nStarting animation... Close the plot window to exit.")
    plt.show()

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

    # Ask user if they want to see the animation
    while True:
        choice = input("\nDo you want to display an animation of this trip? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            try:
                run_animation(all_stars, best_tour, leg_details)
            except Exception as e:
                print(f"\nCould not start animation. Please ensure 'numpy' and 'matplotlib' are installed.")
                print(f"Error: {e}")
            break
        elif choice in ['n', 'no']:
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")