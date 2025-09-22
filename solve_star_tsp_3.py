import csv
import math
import time
import pathlib
from typing import Dict, Optional
from dataclasses import dataclass

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
KM_PER_LY = METERS_PER_LIGHT_YEAR / 1000
ARCSECONDS_PER_DEGREE = 3600.0

# --- Animation Constants ---
TRAIL_LENGTH = 150  # Number of points in the ship's trail
ANIMATION_INTERVAL_MS = 20  # Milliseconds between frames (50 FPS)
SIM_YEARS_PER_SECOND = 10.0 # How many years of sim time pass per second of animation

# --- Simulation Parameters ---
DEFAULT_ACCELERATION_G = 1.0
MIN_ACCELERATION_G = 0.5
MAX_ACCELERATION_G = 100.0

@dataclass
class Star:
    """A dataclass to hold all data for a single star."""
    name: str
    coords: tuple[float, float, float]
    distance_ly: float
    ra_deg: float
    dec_deg: float
    ra_motion_mas_yr: float
    dec_motion_mas_yr: float
    rad_v_km_s: float

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

def load_star_data(filename: str = "closest_stars.csv") -> Dict[str, Star]:
    """
    Loads star data from a CSV and converts spherical coordinates to Cartesian.

    The CSV is expected to have 'star_name', 'distance_ly', 'ra_hms', 'dec_deg',
    'RA_Motion', 'Dec_Motion', and 'Rad_v(km/s)'.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        A dictionary mapping star names to their corresponding Star dataclass
        objects.
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

                stars[star_name] = Star(
                    name=star_name,
                    coords=(x, y, z),
                    distance_ly=dist_ly,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    ra_motion_mas_yr=ra_motion,
                    dec_motion_mas_yr=dec_motion,
                    rad_v_km_s=rad_v_km_s,
                )
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {row} - {e}")
    return stars

def predict_star_position(star_data: Star, time_years: float) -> tuple[float, float, float]:
    """
    Predicts a star's future Cartesian coordinates based on its proper motion
    and radial velocity, as observed from the Sol reference frame.

    Args:
        star_data (Star): The Star object.
        time_years (float): The number of years into the future from t=0.

    Returns:
        A tuple (x, y, z) of the star's predicted coordinates in lightyears.
    """
    # If time is zero, or for Sol, return initial coordinates
    if time_years == 0 or star_data.distance_ly == 0:
        return star_data.coords

    # 1. Calculate change in distance due to radial velocity
    rad_v_ly_per_year = star_data.rad_v_km_s * (SECONDS_PER_YEAR / KM_PER_LY)
    new_dist_ly = star_data.distance_ly + (rad_v_ly_per_year * time_years)

    # 2. Calculate change in RA and Dec due to proper motion
    initial_ra_deg = star_data.ra_deg
    initial_dec_deg = star_data.dec_deg

    # Convert motion from milliarcseconds/year to degrees/year
    ra_motion_deg_per_year = star_data.ra_motion_mas_yr / (1000 * ARCSECONDS_PER_DEGREE)
    dec_motion_deg_per_year = star_data.dec_motion_mas_yr / (1000 * ARCSECONDS_PER_DEGREE)

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
    return math.hypot(
        coords1[0] - coords2[0],
        coords1[1] - coords2[1],
        coords1[2] - coords2[2]
    )

def calculate_interception_leg_details(star1_data: Star, star2_data: Star, departure_sol_time: float, acceleration_g: float):
    """
    Calculates leg details for intercepting a moving target star.

    Args:
        star1_data (Star): Data for the departure star.
        star2_data (Star): Data for the destination star.
        departure_sol_time (float): The time in Sol's frame when the leg begins.
        acceleration_g (float): The ship's acceleration in Gs.

    Returns:
        A dictionary of leg metrics from calculate_leg_details.
    """
    # Position of departure star AT THE TIME OF DEPARTURE
    coords1 = predict_star_position(star1_data, departure_sol_time)

    # --- Iterative estimation to find the interception point ---
    # 1. First guess: travel time to star2's position at the time of departure.
    coords2_guess = predict_star_position(star2_data, departure_sol_time)
    dist_guess = calculate_distance(coords1, coords2_guess)
    _, sol_time_guess = calculate_relativistic_travel_time(dist_guess, acceleration_g)

    # 2. Second, better guess: predict star2's position at the estimated arrival time.
    estimated_arrival_time = departure_sol_time + sol_time_guess
    coords2_final = predict_star_position(star2_data, estimated_arrival_time)

    # 3. Final calculation for the leg using the interception point.
    final_distance = calculate_distance(coords1, coords2_final)
    return calculate_leg_details(final_distance, acceleration_g)

def calculate_relativistic_travel_time(distance_ly: float, acceleration_g: float):
    """
    Calculates ship time (proper time) and Sol time (coordinate time) for a
    trip with constant acceleration for the first half and deceleration
    for the second half.
    Args:
        distance_ly (float): The distance of the travel leg in lightyears.
        acceleration_g (float): The ship's acceleration in Gs.

    Returns:
        A tuple containing (ship_time_years, sol_time_years).
    """
    # Convert distance from lightyears to meters
    d_meters = distance_ly * METERS_PER_LIGHT_YEAR

    # If distance is zero, travel time is zero
    if d_meters == 0:
        return 0.0, 0.0

    # Use shorter variable names for the physics equations
    a = acceleration_g * G_ACCELERATION
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

def calculate_leg_details(distance_ly: float, acceleration_g: float):
    """
    Calculates all travel metrics for a single leg of the journey.

    Args:
        distance_ly (float): The distance of the travel leg in lightyears.
        acceleration_g (float): The ship's acceleration in Gs.

    Returns:
        A dictionary containing detailed metrics for the leg.
    """
    ship_time_years, sol_time_years = calculate_relativistic_travel_time(distance_ly, acceleration_g)

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
    a_ms2 = acceleration_g * G_ACCELERATION
    tanh_arg = (a_ms2 * ship_time_half_sec) / C_METERS_PER_SECOND
    max_speed_fraction_c = math.tanh(tanh_arg)
    max_speed_percent_c = max_speed_fraction_c * 100

    return {
        'distance_ly': distance_ly,
        'ship_time_years': ship_time_years,
        'sol_time_years': sol_time_years,
        'max_speed_percent_c': max_speed_percent_c,
        'time_dilation_factor': time_dilation_factor
    }

def print_tour_summary(tour, leg_details):
    """
    Prints a formatted summary of the calculated TSP tour, including detailed leg data.
    """
    if len(tour) - 1 != len(leg_details):
        print("Warning: Mismatch between tour legs and leg details data.")
        return

    print("\n--- Traveling Salesman Star Tour ---")
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
        sol_time_str = f"{details['sol_time_years']:.6f}"
        max_speed_str = f"{details['max_speed_percent_c']:.6f}"
        dilation_str = f"{details['time_dilation_factor']:.2f}"

        row = (
            f"{leg_name:<38s} | {dist_str:>15s} | {ship_time_str:>15s} | "
            f"{sol_time_str:>15s} | {max_speed_str:>15s} | {dilation_str:>15s}"
        )
        print(row)

def solve_tdtsp_nearest_neighbor(stars: Dict[str, Star], start_star: str, acceleration_g: float):
    """
    Solves the Time-Dependent TSP using a Nearest Neighbor heuristic.

    At each step, it chooses the next star that is "closest in time" to reach,
    accounting for stellar motion.

    Args:
        stars (Dict[str, Star]): A dictionary of star data from load_star_data.
        start_star (str): The name of the starting star.
        acceleration_g (float): The ship's acceleration in Gs.

    Returns:
        A tuple containing the tour, total ship time, total Sol time, and a
        list of detailed leg metrics.
    """
    if start_star not in stars:
        raise ValueError("Start star not found in the dataset.")

    unvisited = set(stars.keys())
    tour = [start_star]
    unvisited.remove(start_star)

    current_star_name = start_star
    total_ship_time = 0.0
    total_sol_time = 0.0
    cumulative_sol_time = 0.0
    leg_details_list = []

    while unvisited:
        best_next_star = None
        best_leg_details = None
        min_sol_time_leg = float('inf')

        star1_data = stars[current_star_name]

        # Find the unvisited star that is closest in travel time from the current location and time
        for next_star_name in unvisited:
            star2_data = stars[next_star_name]
            leg_details = calculate_interception_leg_details(star1_data, star2_data, cumulative_sol_time, acceleration_g)

            if leg_details['sol_time_years'] < min_sol_time_leg:
                min_sol_time_leg = leg_details['sol_time_years']
                best_next_star = next_star_name
                best_leg_details = leg_details

        # Add the best leg found to our tour totals
        leg_details_list.append(best_leg_details)
        total_ship_time += best_leg_details['ship_time_years']
        total_sol_time += best_leg_details['sol_time_years']
        cumulative_sol_time += best_leg_details['sol_time_years']

        # Add 1-year rest time and update cumulative time
        rest_time = 1.0
        total_ship_time += rest_time
        total_sol_time += rest_time
        cumulative_sol_time += rest_time

        # Move to the next star
        current_star_name = best_next_star
        tour.append(current_star_name)
        unvisited.remove(current_star_name)

    # The last leg is from the final star back to Sol
    final_leg_details = calculate_interception_leg_details(stars[current_star_name], stars[start_star], cumulative_sol_time, acceleration_g)
    leg_details_list.append(final_leg_details)
    total_ship_time += final_leg_details['ship_time_years']
    total_sol_time += final_leg_details['sol_time_years']
    tour.append(start_star)

    # We added one rest period too many in the loop (for the last stop before returning to Sol)
    total_ship_time -= rest_time
    total_sol_time -= rest_time

    return tour, total_ship_time, total_sol_time, leg_details_list

def print_execution_time(start_time, end_time):
    """
    Calculates and prints the script's total execution time in milliseconds.
    """
    duration_s = end_time - start_time
    duration_ms = duration_s * 1000
    print("\n--- Script Execution Time ---")
    print(f"Total time: {duration_ms:.3f} milliseconds")

def get_ship_position_at_sol_time(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    total_sol_time_leg: float,
    time_into_leg_years: float,
    acceleration_g: float
) -> np.ndarray:
    """
    Calculates the ship's 3D position at a specific coordinate time (Sol time)
    within a leg, assuming constant proper acceleration/deceleration.
    """
    direction_vector = end_pos - start_pos
    total_distance_ly = np.linalg.norm(direction_vector)

    if total_distance_ly == 0 or total_sol_time_leg == 0:
        return start_pos

    unit_direction = direction_vector / total_distance_ly

    # Acceleration in ly/year^2
    a_ms2 = acceleration_g * G_ACCELERATION
    a_ly_per_year_sq = a_ms2 * (SECONDS_PER_YEAR**2) / METERS_PER_LIGHT_YEAR
    c_ly_per_year = 1.0  # By definition

    def dist_from_rest(t_years: float) -> float:
        """
        Calculates relativistic distance traveled from rest at coordinate time t.
        """
        if t_years <= 0:
            return 0.0
        term = (a_ly_per_year_sq * t_years)**2
        dist_ly = (c_ly_per_year**2 / a_ly_per_year_sq) * (np.sqrt(1 + term) - 1)
        return dist_ly

    midpoint_sol_time = total_sol_time_leg / 2.0

    if time_into_leg_years <= midpoint_sol_time:
        distance_traveled = dist_from_rest(time_into_leg_years)
    else:
        time_from_end = total_sol_time_leg - time_into_leg_years
        distance_from_end = dist_from_rest(time_from_end)
        distance_traveled = total_distance_ly - distance_from_end

    distance_traveled = np.clip(distance_traveled, 0, total_distance_ly)
    current_pos = start_pos + distance_traveled * unit_direction
    return current_pos

def run_animation(all_stars: Dict[str, Star], tour: list, leg_details: list, acceleration_g: float):
    """Sets up and runs the Matplotlib 3D tour animation."""
    # --- 1. Pre-calculate the full tour schedule for animation ---
    tour_schedule = []
    cumulative_sol_time = 0.0
    for i, leg in enumerate(leg_details):
        start_star_name = tour[i]
        end_star_name = tour[i+1]

        start_pos = np.array(predict_star_position(all_stars[start_star_name], cumulative_sol_time))
        leg_sol_time = leg['sol_time_years']
        arrival_sol_time = cumulative_sol_time + leg_sol_time
        end_pos = np.array(predict_star_position(all_stars[end_star_name], arrival_sol_time))

        tour_schedule.append({
            'start_time': cumulative_sol_time,
            'end_time': arrival_sol_time,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'duration': leg_sol_time,
        })

        cumulative_sol_time = arrival_sol_time
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

    star_coords = np.array([s.coords for s in all_stars.values()])
    ax.scatter(star_coords[:, 0], star_coords[:, 1], star_coords[:, 2], s=10, c='white', alpha=0.7)
    tour_star_names = set(tour)
    for name, star in all_stars.items():
        if name in tour_star_names:
            ax.text(star.coords[0], star.coords[1], star.coords[2], f' {name}', color='cyan', fontsize=9)

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
            ship_pos = get_ship_position_at_sol_time(
                current_leg['start_pos'], current_leg['end_pos'], current_leg['duration'], time_into_leg, acceleration_g
            )
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

    # Get acceleration input from the user
    acceleration_g = 0.0
    while True:
        try:
            prompt = (
                f"Enter ship's constant acceleration in Gs "
                f"({MIN_ACCELERATION_G} to {MAX_ACCELERATION_G}) [{DEFAULT_ACCELERATION_G}]: "
            )
            accel_input = input(prompt)
            if not accel_input: # Default to 1.0 if user just presses Enter
                acceleration_g = DEFAULT_ACCELERATION_G
                break
            acceleration_g = float(accel_input)
            if MIN_ACCELERATION_G <= acceleration_g <= MAX_ACCELERATION_G:
                break
            else:
                print(
                    f"Error: Acceleration must be between "
                    f"{MIN_ACCELERATION_G} and {MAX_ACCELERATION_G} Gs."
                )
        except ValueError:
            print("Error: Please enter a valid number.")

    # 2. Define the starting point and solve the TSP
    print(f"\nCalculating shortest tour from {args.start_star} using Time-Dependent Nearest Neighbor heuristic...")
    try:
        tour, ship_time, sol_time, leg_details = solve_tdtsp_nearest_neighbor(all_stars, args.start_star, acceleration_g)

        # Print the detailed tour summary using the leg details
        print_tour_summary(tour, leg_details)

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
        print(f"Assumptions: {acceleration_g}g constant accel/decel, 1-year stay at each destination.")
        print(f"Total subjective time on ship (travel + stops): {ship_time:.2f} years")
        print(f"Total time in Sol's frame (travel + stops):   {sol_time:.2f} years")
        print(f"Average time dilation factor (travel only):     {average_dilation:.2f}x")
        print(f"Average max speed (weighted by leg duration):   {weighted_avg_max_speed:.6f}% of c")

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
                run_animation(all_stars, tour, leg_details, acceleration_g)
            except NameError: # If tour failed, variables won't be defined
                print("\nCannot start animation because tour calculation failed.")
            except Exception as e:
                print(f"\nCould not start animation. Please ensure 'numpy' and 'matplotlib' are installed.")
                print(f"Error: {e}")
            break
        elif choice in ['n', 'no']:
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")