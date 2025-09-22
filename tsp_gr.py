import csv
import math
import time
import pathlib
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Physical Constants ---
C_METERS_PER_SECOND = 299792458.0
G_ACCELERATION = 9.80665
SECONDS_PER_YEAR = 31557600.0
METERS_PER_LIGHT_YEAR = C_METERS_PER_SECOND * SECONDS_PER_YEAR
KM_PER_LY = METERS_PER_LIGHT_YEAR / 1000
ARCSECONDS_PER_DEGREE = 3600.0
G_GRAVITATIONAL_CONSTANT = 6.67430e-11
AU_METERS = 149597870700.0
SOL_MASS_KG = 1.989e30

# --- Simulation Parameters ---
DEFAULT_ACCELERATION_G = 1.0
MIN_ACCELERATION_G = 0.5
MAX_ACCELERATION_G = 100.0
REST_TIME_YEARS = 1.0
NUM_THREADS = 6
LEG_CALCULATION_TIMEOUT_S = 1800 # 30 minutes

# --- Animation Constants ---
TRAIL_LENGTH = 150
ANIMATION_INTERVAL_MS = 20
SIM_YEARS_PER_SECOND = 10.0

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
    mass_sol: float

def hms_to_decimal(hms_str: str) -> Optional[float]:
    """Converts a Right Ascension string (hh:mm:ss.s) to decimal degrees."""
    if hms_str.strip() == '0':
        return 0.0
    try:
        hours, minutes, seconds = map(float, hms_str.split(':'))
        if not (0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60):
            return None
        return 15 * (hours + minutes / 60 + seconds / 3600)
    except (ValueError, TypeError):
        return None

def load_star_data(filename: str) -> Dict[str, Star]:
    """Loads star data from a CSV, including the new mass column."""
    stars = {}
    with open(filename, mode='r', encoding='utf-8-sig', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            try:
                star_name = row['star_name']
                dist_ly = float(row['distance_ly'])
                ra_deg = hms_to_decimal(row['ra_hms'])
                dec_deg = float(row['dec_deg'])
                ra_motion = float(row.get('RA_Motion', '0'))
                dec_motion = float(row.get('Dec_Motion', '0'))
                rad_v_km_s = float(row.get('Rad_v(km/s)', '0'))
                mass_sol = float(row.get('mass', '0'))

                if ra_deg is None:
                    print(f"Skipping star {star_name} due to invalid RA format.")
                    continue

                ra_rad = math.radians(ra_deg)
                dec_rad = math.radians(dec_deg)
                x = dist_ly * math.cos(dec_rad) * math.cos(ra_rad)
                y = dist_ly * math.cos(dec_rad) * math.sin(ra_rad)
                z = dist_ly * math.sin(dec_rad)

                stars[star_name] = Star(
                    name=star_name, coords=(x, y, z), distance_ly=dist_ly,
                    ra_deg=ra_deg, dec_deg=dec_deg, ra_motion_mas_yr=ra_motion,
                    dec_motion_mas_yr=dec_motion, rad_v_km_s=rad_v_km_s, mass_sol=mass_sol
                )
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {row} - {e}")
    return stars

def predict_star_position(star_data: Star, time_years: float) -> np.ndarray:
    """Predicts a star's future Cartesian coordinates in lightyears."""
    if time_years == 0 or star_data.distance_ly == 0:
        return np.array(star_data.coords)

    rad_v_ly_per_year = star_data.rad_v_km_s * (SECONDS_PER_YEAR / KM_PER_LY)
    new_dist_ly = star_data.distance_ly + (rad_v_ly_per_year * time_years)

    ra_motion_deg_yr = star_data.ra_motion_mas_yr / (1000 * ARCSECONDS_PER_DEGREE)
    dec_motion_deg_yr = star_data.dec_motion_mas_yr / (1000 * ARCSECONDS_PER_DEGREE)

    new_ra_deg = star_data.ra_deg + (ra_motion_deg_yr * time_years)
    new_dec_deg = star_data.dec_deg + (dec_motion_deg_yr * time_years)

    new_ra_rad = math.radians(new_ra_deg)
    new_dec_rad = math.radians(new_dec_deg)

    x = new_dist_ly * math.cos(new_dec_rad) * math.cos(new_ra_rad)
    y = new_dist_ly * math.cos(new_dec_rad) * math.sin(new_ra_rad)
    z = new_dist_ly * math.sin(new_dec_rad)

    return np.array([x, y, z])

# --- General Relativity Physics Core ---

def get_gravity_at_point(ship_pos_m: np.ndarray, time_years: float, all_stars: Dict[str, Star], exclude_star_name: str = None) -> np.ndarray:
    """
    Calculates the total Newtonian gravitational acceleration vector at a point in space.
    This serves as a first-order approximation for the effects of curved spacetime.
    """
    total_accel_m_s2 = np.zeros(3)
    for star in all_stars.values():
        if star.mass_sol == 0:
            continue

        # If we are inside the orbital radius of the star we are trying to leave,
        # assume its gravity has been handled by the orbital burn and ignore it
        # to prevent a singularity-like effect at the start of the integration.
        if star.name == exclude_star_name:
            orbital_radius_m = star.mass_sol * AU_METERS
            if np.sum((predict_star_position(star, time_years) * METERS_PER_LIGHT_YEAR - ship_pos_m)**2) < orbital_radius_m**2:
                continue

        star_pos_ly = predict_star_position(star, time_years)
        star_pos_m = star_pos_ly * METERS_PER_LIGHT_YEAR

        r_vec = star_pos_m - ship_pos_m
        dist_sq = np.sum(r_vec**2)
        if dist_sq == 0:
            continue
        
        dist = np.sqrt(dist_sq)
        force_dir = r_vec / dist
        
        star_mass_kg = star.mass_sol * SOL_MASS_KG
        accel_magnitude = G_GRAVITATIONAL_CONSTANT * star_mass_kg / dist_sq
        total_accel_m_s2 += accel_magnitude * force_dir
        
    return total_accel_m_s2

def calculate_orbital_maneuver(star: Star, interstellar_velocity_ms: float, acceleration_g: float) -> tuple[float, float]:
    """
    Calculates the delta-v and time required for an orbital maneuver.
    This covers both escape from orbit and insertion into orbit.
    """
    if star.mass_sol == 0:
        return 0.0, 0.0

    # Assumption: The planet is at a distance proportional to the star's mass.
    orbital_radius_m = star.mass_sol * AU_METERS
    star_mass_kg = star.mass_sol * SOL_MASS_KG

    # Velocity required for a circular orbit at this radius
    v_orbit_ms = np.sqrt(G_GRAVITATIONAL_CONSTANT * star_mass_kg / orbital_radius_m)
    
    # Escape velocity from that orbital radius
    v_escape_ms = np.sqrt(2 * G_GRAVITATIONAL_CONSTANT * star_mass_kg / orbital_radius_m)

    # Using the Oberth effect, calculate the velocity required at periapsis
    # of the hyperbolic trajectory. v_inf is the final interstellar velocity.
    v_post_burn_ms = np.sqrt(interstellar_velocity_ms**2 + v_escape_ms**2)
    
    # The delta-v is the difference between the velocity in low orbit and the velocity
    # needed at that same altitude to achieve the desired hyperbolic trajectory.
    delta_v_ms = v_post_burn_ms - v_orbit_ms
    
    # Time to perform the burn at the ship's constant acceleration
    burn_time_s = delta_v_ms / (acceleration_g * G_ACCELERATION)
    
    return delta_v_ms, burn_time_s

def sr_travel_time(distance_ly: float, acceleration_g: float) -> float:
    """
    A helper function to get a quick Special Relativity-based estimate for travel time.
    Used to provide a reasonable time-span for the GR numerical integration.
    """
    d_meters = distance_ly * METERS_PER_LIGHT_YEAR
    if d_meters == 0: return 0.0
    a = acceleration_g * G_ACCELERATION
    c = C_METERS_PER_SECOND
    term = (a * d_meters) / (2 * c**2)
    sol_time_leg_sec = 2 * (c / a) * np.sqrt((1 + term)**2 - 1)
    return sol_time_leg_sec / SECONDS_PER_YEAR

def ode_system_for_scipy(t, y, all_stars, target_star, acceleration_g, t_start_years, t_mid_years, start_star_name):
    """
    Defines the system of Ordinary Differential Equations for the ship's motion.
    This function is called by the numerical integrator (solve_ivp).
    
    y = [x, y, z, vx, vy, vz, proper_time]
    """
    ship_pos_m = y[:3]
    ship_vel_ms = y[3:6]
    
    current_time_years = t_start_years + (t / SECONDS_PER_YEAR)
    
    # 1. Gravitational Acceleration
    accel_gravity = get_gravity_at_point(ship_pos_m, current_time_years, all_stars, start_star_name)
    
    # 2. Propulsion Acceleration (Heuristic Brachistochrone)
    propulsion_accel_mag = acceleration_g * G_ACCELERATION
    
    # For the first half of the trip, accelerate towards the target.
    # For the second half, decelerate (thrust opposite to velocity).
    if current_time_years < t_mid_years:
        target_pos_ly = predict_star_position(target_star, current_time_years)
        target_pos_m = target_pos_ly * METERS_PER_LIGHT_YEAR
        propulsion_dir = (target_pos_m - ship_pos_m)
        norm = np.linalg.norm(propulsion_dir)
        if norm > 0:
            propulsion_dir /= norm
    else:
        norm = np.linalg.norm(ship_vel_ms)
        if norm > 0:
            propulsion_dir = -ship_vel_ms / norm
        else:
            # If ship is stationary, no propulsion direction for deceleration
            propulsion_dir = np.zeros(3)
        
    accel_propulsion = propulsion_dir * propulsion_accel_mag
    
    # 3. Total acceleration
    total_accel = accel_gravity + accel_propulsion
    
    # 4. Proper time integration (dτ/dt)
    # dτ/dt = sqrt(1 - v²/c²)
    v_sq = np.sum(ship_vel_ms**2)
    lorentz_term = 1 - v_sq / C_METERS_PER_SECOND**2
    if lorentz_term > 0:
        d_proper_time_dt = np.sqrt(lorentz_term)
    else:
        # If v >= c due to numerical error, proper time effectively stops elapsing.
        d_proper_time_dt = 0.0
    
    return [
        ship_vel_ms[0], ship_vel_ms[1], ship_vel_ms[2],
        total_accel[0], total_accel[1], total_accel[2],
        d_proper_time_dt
    ]

def solve_leg_with_gravity(args_tuple):
    """
    Worker function to solve a single leg of the journey using GR approximations.
    Designed to be called by the multiprocessing pool.
    
    This is the core of the "shooting method". It uses a root-finding algorithm
    to find the correct departure angle to hit a moving target in a gravitational field.
    """
    # Unpack arguments
    start_star, end_star, all_stars, departure_sol_time, acceleration_g = args_tuple
    
    # --- Initial Setup ---
    start_pos_ly = predict_star_position(start_star, departure_sol_time)
    
    # Use SR calculation for an initial guess of the leg duration
    # This is crucial for giving the numerical solver a reasonable timeframe.
    end_pos_guess_ly = predict_star_position(end_star, departure_sol_time)
    dist_guess_ly = np.linalg.norm(end_pos_guess_ly - start_pos_ly)
    time_guess_years = sr_travel_time(dist_guess_ly, acceleration_g)
    
    if time_guess_years < 1e-3: # Effectively zero distance
        return {'sol_time_years': 0, 'ship_time_years': 0, 'start_star': start_star.name, 'end_star': end_star.name}

    # The midpoint of the journey in absolute time
    t_mid_years = departure_sol_time + (time_guess_years / 2.0)

    # --- The "Shooting Method" ---
    # We need to find the correct initial velocity vector to hit the target.
    # We parameterize this vector with two angles (theta, phi) and a magnitude.
    # The magnitude is a rough guess based on SR.
    v_mag_guess = (dist_guess_ly * METERS_PER_LIGHT_YEAR) / (time_guess_years * SECONDS_PER_YEAR)

    def residual(initial_velocity_vector):
        """
        This is the function that scipy.optimize.root will try to drive to zero.
        It simulates a journey with a given departure velocity and returns the "miss distance".
        """
        vx, vy, vz = initial_velocity_vector
        
        y0 = np.array([
            start_pos_ly[0] * METERS_PER_LIGHT_YEAR,
            start_pos_ly[1] * METERS_PER_LIGHT_YEAR,
            start_pos_ly[2] * METERS_PER_LIGHT_YEAR,
            vx, vy, vz, 0.0 # Initial proper time is 0
        ])
        
        t_span_s = [0, time_guess_years * SECONDS_PER_YEAR * 2.5] # Integrate for 250% of guessed time
        
        # Run the numerical integration
        sol = solve_ivp(
            ode_system_for_scipy,
            t_span_s,
            y0,
            args=(all_stars, end_star, acceleration_g, departure_sol_time, t_mid_years, start_star.name),
            dense_output=True,
            method='DOP853' # A good, high-precision solver
        )
        
        # Final position of the ship
        final_ship_pos_m = sol.y[:3, -1]
        
        # Where the target star is at that final time
        final_time_years = departure_sol_time + (sol.t[-1] / SECONDS_PER_YEAR)
        final_target_pos_m = predict_star_position(end_star, final_time_years) * METERS_PER_LIGHT_YEAR
        
        # The "miss distance" vector. Root finder will minimize this.
        miss_vector = final_ship_pos_m - final_target_pos_m
        return miss_vector

    # Initial guess for the velocity vector: point directly at the target's initial position
    # with a magnitude based on the SR average velocity.
    initial_dir = end_pos_guess_ly - start_pos_ly
    norm = np.linalg.norm(initial_dir)
    if norm > 0:
        initial_dir /= norm
    initial_velocity_guess = initial_dir * v_mag_guess
    
    # Suppress warnings from the solver, which can be noisy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Find the correct initial velocity vector that results in a hit
        solution = root(residual, initial_velocity_guess, method='hybr', tol=100000000.0) # Tol in meters (100,000 km)

    if not solution.success:
        # If the solver fails, it's a very difficult shot. Return infinity.
        return {'sol_time_years': float('inf'), 'ship_time_years': float('inf'), 'start_star': start_star.name, 'end_star': end_star.name}

    # --- Final Calculation with Correct Path ---
    # Now that we have the correct angles, run the simulation one last time to get the final data.
    final_velocity = solution.x

    y0 = np.array([
        start_pos_ly[0] * METERS_PER_LIGHT_YEAR, start_pos_ly[1] * METERS_PER_LIGHT_YEAR, start_pos_ly[2] * METERS_PER_LIGHT_YEAR,
        final_velocity[0], final_velocity[1], final_velocity[2], 0.0
    ])
    t_span_s = [0, time_guess_years * SECONDS_PER_YEAR * 2.5]
    
    sol = solve_ivp(
        ode_system_for_scipy, t_span_s, y0,
        args=(all_stars, end_star, acceleration_g, departure_sol_time, t_mid_years, start_star.name),
        dense_output=True, method='DOP853'
    )

    sol_time_s = sol.t[-1]
    ship_time_s = sol.y[6, -1]
    
    # --- Orbital Maneuvers ---
    # Calculate time for departure and arrival burns
    initial_interstellar_vel = np.linalg.norm(y0[3:6])
    final_interstellar_vel = np.linalg.norm(sol.y[3:6, -1])
    
    _, departure_burn_time_s = calculate_orbital_maneuver(start_star, initial_interstellar_vel, acceleration_g)
    _, arrival_burn_time_s = calculate_orbital_maneuver(end_star, final_interstellar_vel, acceleration_g)
    
    total_burn_time_s = departure_burn_time_s + arrival_burn_time_s

    # Add burn times to total leg times. Assume they are short enough that
    # time dilation is negligible during the burn itself.
    total_sol_time_s = sol_time_s + total_burn_time_s
    total_ship_time_s = ship_time_s + total_burn_time_s

    # Final leg details
    final_ship_pos_m = sol.y[:3, -1]
    final_target_pos_m = predict_star_position(end_star, departure_sol_time + total_sol_time_s / SECONDS_PER_YEAR) * METERS_PER_LIGHT_YEAR
    distance_m = np.linalg.norm(final_target_pos_m - (y0[:3]))

    max_speed_ms = np.max(np.linalg.norm(sol.y[3:6, :], axis=0))

    return {
        'sol_time_years': total_sol_time_s / SECONDS_PER_YEAR,
        'ship_time_years': total_ship_time_s / SECONDS_PER_YEAR,
        'distance_ly': distance_m / METERS_PER_LIGHT_YEAR,
        'max_speed_percent_c': (max_speed_ms / C_METERS_PER_SECOND) * 100,
        'time_dilation_factor': total_sol_time_s / total_ship_time_s if total_ship_time_s > 0 else 1,
        'start_star': start_star.name,
        'end_star': end_star.name
    }

def solve_gr_tsp_nearest_neighbor(stars: Dict[str, Star], start_star: str, acceleration_g: float):
    """
    Solves the TSP using the GR-aware Nearest Neighbor heuristic.
    Uses a process pool to parallelize the computationally expensive leg calculations.
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

    print("\nStarting GR Tour Calculation. This will take some time...")
    print(f"Using {NUM_THREADS} threads for parallel computation.")

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        while unvisited:
            print(f"\nCalculating legs from: {current_star_name}. Remaining: {len(unvisited)}")
            
            # Create a list of tasks for the process pool
            tasks = []
            for next_star_name in unvisited:
                args = (stars[current_star_name], stars[next_star_name], stars, cumulative_sol_time, acceleration_g)
                tasks.append(executor.submit(solve_leg_with_gravity, args))

            # Process results as they complete
            results = []
            for future in as_completed(tasks):
                try:
                    # Add a timeout to prevent the script from hanging on a single difficult calculation
                    result = future.result(timeout=LEG_CALCULATION_TIMEOUT_S)
                    results.append(result)
                    print(f"  > Completed leg: {result['start_star']} -> {result['end_star']} (Sol Time: {result['sol_time_years']:.2f} y)")
                except TimeoutError:
                    print(f"  > A leg calculation timed out after {LEG_CALCULATION_TIMEOUT_S} seconds.")
                except Exception as e:
                    print(f"  > A leg calculation failed: {e}")

            # Find the best leg from the completed results
            if not results:
                raise RuntimeError("All leg calculations failed. Cannot continue tour.")
                
            best_leg_details = min(results, key=lambda r: r['sol_time_years'])
            best_next_star = best_leg_details['end_star']
            
            # Add the best leg found to our tour totals
            leg_details_list.append(best_leg_details)
            total_ship_time += best_leg_details['ship_time_years']
            total_sol_time += best_leg_details['sol_time_years']
            cumulative_sol_time += best_leg_details['sol_time_years']

            # Add rest time
            total_ship_time += REST_TIME_YEARS
            total_sol_time += REST_TIME_YEARS
            cumulative_sol_time += REST_TIME_YEARS

            # Move to the next star
            current_star_name = best_next_star
            tour.append(current_star_name)
            unvisited.remove(current_star_name)

        # --- Final leg back to Sol ---
        print(f"\nCalculating final leg from: {current_star_name} -> {start_star}")
        args = (stars[current_star_name], stars[start_star], stars, cumulative_sol_time, acceleration_g)
        final_leg_details = solve_leg_with_gravity(args)
        print(f"  > Completed leg: {final_leg_details['start_star']} -> {final_leg_details['end_star']} (Sol Time: {final_leg_details['sol_time_years']:.2f} y)")

    leg_details_list.append(final_leg_details)
    total_ship_time += final_leg_details['ship_time_years']
    total_sol_time += final_leg_details['sol_time_years']
    tour.append(start_star)

    # We added one rest period too many
    total_ship_time -= REST_TIME_YEARS
    total_sol_time -= REST_TIME_YEARS

    return tour, total_ship_time, total_sol_time, leg_details_list

def print_tour_summary(tour, leg_details):
    """Prints a formatted summary of the calculated tour."""
    print("\n--- Traveling Salesman Star Tour (GR-Aware) ---")
    print("Tour path: " + " -> ".join(tour))
    print("\n--- Leg-by-Leg Analysis ---")
    header = (f"{'Leg':<38s} | {'Distance (ly)':>15s} | {'Ship Time (y)':>15s} | "
              f"{'Sol Time (y)':>15s} | {'Max Speed (%c)':>15s} | {'Time Dilation':>15s}")
    print(header)
    print("-" * len(header))

    for details in leg_details:
        leg_name = f"{details['start_star']} -> {details['end_star']}"
        dist_str = f"{details['distance_ly']:.2f}"
        ship_time_str = f"{details['ship_time_years']:.2f}"
        sol_time_str = f"{details['sol_time_years']:.6f}"
        max_speed_str = f"{details['max_speed_percent_c']:.6f}"
        dilation_str = f"{details['time_dilation_factor']:.2f}"
        row = (f"{leg_name:<38s} | {dist_str:>15s} | {ship_time_str:>15s} | "
               f"{sol_time_str:>15s} | {max_speed_str:>15s} | {dilation_str:>15s}")
        print(row)

def print_execution_time(start_time, end_time):
    """Calculates and prints the script's total execution time."""
    duration_s = end_time - start_time
    print("\n--- Script Execution Time ---")
    if duration_s > 60:
        minutes, seconds = divmod(duration_s, 60)
        print(f"Total time: {int(minutes)} minutes, {seconds:.3f} seconds")
    else:
        print(f"Total time: {duration_s:.3f} seconds")

# NOTE: Animation functions are omitted for brevity in this GR version,
# as the focus is on the complex calculation core. They could be re-added
# by storing the `sol.y` path from the final integration of each leg.

if __name__ == "__main__":
    # This check is crucial for multiprocessing on Windows
    # It prevents child processes from re-executing the main script block.
    
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(description="Solve a GR-aware TSP for a list of stars.")
    parser.add_argument("csv_file", help="Path to the CSV file containing star data.")
    parser.add_argument("--start_star", default="Sol", help="The name of the star to start the tour from.")
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv_file)
    if not csv_path.is_absolute():
        script_dir = pathlib.Path(__file__).parent.resolve()
        csv_path = script_dir / csv_path

    print(f"Loading star data from '{csv_path}'...")
    all_stars = load_star_data(csv_path)
    print(f"Loaded data for {len(all_stars)} stars.")

    acceleration_g = 0.0
    while True:
        try:
            prompt = (f"Enter ship's constant acceleration in Gs "
                      f"({MIN_ACCELERATION_G} to {MAX_ACCELERATION_G}) [{DEFAULT_ACCELERATION_G}]: ")
            accel_input = input(prompt)
            if not accel_input:
                acceleration_g = DEFAULT_ACCELERATION_G
                break
            acceleration_g = float(accel_input)
            if MIN_ACCELERATION_G <= acceleration_g <= MAX_ACCELERATION_G:
                break
            else:
                print(f"Error: Acceleration must be between {MIN_ACCELERATION_G} and {MAX_ACCELERATION_G} Gs.")
        except ValueError:
            print("Error: Please enter a valid number.")

    try:
        tour, ship_time, sol_time, leg_details = solve_gr_tsp_nearest_neighbor(all_stars, args.start_star, acceleration_g)

        print_tour_summary(tour, leg_details)

        total_travel_ship_time = sum(d['ship_time_years'] for d in leg_details)
        total_travel_sol_time = sum(d['sol_time_years'] for d in leg_details)
        average_dilation = (total_travel_sol_time / total_travel_ship_time) if total_travel_ship_time > 0 else 1.0
        weighted_avg_max_speed = (sum(d['max_speed_percent_c'] * d['sol_time_years'] for d in leg_details) / total_travel_sol_time) if total_travel_sol_time > 0 else 0.0

        print("\n--- Relativistic Travel Time Analysis (GR-Aware) ---")
        print(f"Assumptions: {acceleration_g}g accel/decel, {REST_TIME_YEARS}-year stay, orbital maneuvers, n-body gravity.")
        print(f"Total subjective time on ship (travel + stops): {ship_time:.2f} years")
        print(f"Total time in Sol's frame (travel + stops):   {sol_time:.2f} years")
        print(f"Average time dilation factor (travel only):     {average_dilation:.2f}x")
        print(f"Average max speed (weighted by leg duration):   {weighted_avg_max_speed:.6f}% of c")

    except (ValueError, RuntimeError) as e:
        print(f"\nAn error occurred during tour calculation: {e}")

    end_time = time.perf_counter()
    print_execution_time(start_time, end_time)

    # Animation prompt is omitted as the animation code is not included in this version.