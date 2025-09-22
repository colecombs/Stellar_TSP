import argparse
import pathlib
import collections
import math

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# This script requires the main solver script to be in the same directory.
# It also requires numpy and matplotlib:
# pip install numpy matplotlib
try:
    import solve_star_tsp_3 as star_tsp
except ImportError:
    print("Error: Could not import 'solve_star_tsp_3.py'.")
    print("Please ensure that 'solve_star_tsp_3.py' is in the same directory as this script.")
    exit(1)

# --- Animation Constants ---
TRAIL_LENGTH = 150  # Number of points in the ship's trail
ANIMATION_INTERVAL_MS = 20  # Milliseconds between frames (50 FPS)
SIM_YEARS_PER_SECOND = 10.0 # How many years of sim time pass per second of animation

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

    Args:
        start_pos (np.ndarray): The 3D coordinates of the departure star.
        end_pos (np.ndarray): The 3D coordinates of the arrival star (interception point).
        total_sol_time_leg (float): The total Sol time for this leg in years.
        time_into_leg_years (float): The elapsed Sol time since the leg began.
        acceleration_g (float): The ship's acceleration in Gs.

    Returns:
        np.ndarray: The calculated 3D position of the ship in lightyears.
    """
    direction_vector = end_pos - start_pos
    total_distance_ly = np.linalg.norm(direction_vector)

    if total_distance_ly == 0 or total_sol_time_leg == 0:
        return start_pos

    unit_direction = direction_vector / total_distance_ly

    # Acceleration in ly/year^2
    a_ms2 = acceleration_g * star_tsp.G_ACCELERATION
    a_ly_per_year_sq = a_ms2 * (star_tsp.SECONDS_PER_YEAR**2) / star_tsp.METERS_PER_LIGHT_YEAR
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

def main():
    """Main function to set up and run the visualization."""
    # Start the high-resolution performance counter
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Visualize a relativistic star tour calculated by solve_star_tsp_3.py."
    )
    parser.add_argument("csv_file", help="Path to the CSV file containing star data.")
    parser.add_argument(
        "--start_star", default="Sol", help="The name of the star to start the tour from (default: Sol)."
    )
    args = parser.parse_args()

    # --- 1. Load data and solve tour using the imported script ---
    csv_path = pathlib.Path(args.csv_file)
    if not csv_path.is_absolute():
        script_dir = pathlib.Path(__file__).parent.resolve()
        csv_path = script_dir / csv_path

    print(f"Loading star data from '{csv_path}'...")
    all_stars = star_tsp.load_star_data(csv_path)
    print(f"Loaded data for {len(all_stars)} stars.")

    # Get acceleration input from the user
    acceleration_g = 0.0
    while True:
        try:
            accel_input = input("Enter ship's constant acceleration in Gs (0.5 to 10.0) [1.0]: ")
            if not accel_input: # Default to 1.0 if user just presses Enter
                acceleration_g = 1.0
                break
            acceleration_g = float(accel_input)
            if 0.5 <= acceleration_g <= 10.0:
                break
            else:
                print("Error: Acceleration must be between 0.5 and 10.0 Gs.")
        except ValueError:
            print("Error: Please enter a valid number.")

    print(f"\nCalculating tour from {args.start_star}...")
    try:
        tour, ship_time, sol_time, leg_details = star_tsp.solve_tdtsp_nearest_neighbor(all_stars, args.start_star, acceleration_g)

        # --- Display the same summary as the main script ---
        star_tsp.print_tour_summary(tour, leg_details)

        total_travel_ship_time = sum(d['ship_time_years'] for d in leg_details)
        total_travel_sol_time = sum(d['sol_time_years'] for d in leg_details)
        average_dilation = (total_travel_sol_time / total_travel_ship_time) if total_travel_ship_time > 0 else 1.0
        weighted_avg_max_speed = (sum(d['max_speed_percent_c'] * d['sol_time_years'] for d in leg_details) / total_travel_sol_time) if total_travel_sol_time > 0 else 0.0

        print("\n--- Relativistic Travel Time Analysis ---")
        print(f"Assumptions: {acceleration_g}g constant accel/decel, 1-year stay at each destination.")
        print(f"Total subjective time on ship (travel + stops): {ship_time:.2f} years")
        print(f"Total time in Sol's frame (travel + stops):   {sol_time:.2f} years")
        print(f"Average time dilation factor (travel only):     {average_dilation:.2f}x")
        print(f"Average max speed (weighted by leg duration):   {weighted_avg_max_speed:.2f}% of c")

    except ValueError as e:
        print(f"\nError: {e}")
        # Stop the counter and print the elapsed time on error
        end_time = time.perf_counter()
        star_tsp.print_execution_time(start_time, end_time)
        return # Exit if tour calculation fails

    print("\nTour calculation and summary complete. Preparing visualization...")
    # --- 2. Pre-calculate the full tour schedule for animation ---
    # This determines the exact start/end coordinates for each leg based on stellar motion.
    tour_schedule = []
    cumulative_sol_time = 0.0
    for i, leg in enumerate(leg_details):
        start_star_name = tour[i]
        end_star_name = tour[i+1]

        # Position of departure star AT THE TIME OF DEPARTURE
        start_pos = np.array(star_tsp.predict_star_position(all_stars[start_star_name], cumulative_sol_time))

        leg_sol_time = leg['sol_time_years']
        arrival_sol_time = cumulative_sol_time + leg_sol_time

        # Position of arrival star AT THE TIME OF ARRIVAL (interception point)
        end_pos = np.array(star_tsp.predict_star_position(all_stars[end_star_name], arrival_sol_time))

        tour_schedule.append({
            'leg_name': f"{start_star_name} -> {end_star_name}",
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

    # --- 3. Set up the Matplotlib 3D plot ---
    fig = plt.figure(figsize=(12, 12), facecolor='black')
    ax = fig.add_subplot(projection='3d', facecolor='black')
    ax.set_title("Relativistic Star Tour", color='white')
    ax.set_xlabel("X (ly)", color='white')
    ax.set_ylabel("Y (ly)", color='white')
    ax.set_zlabel("Z (ly)", color='white')

    # Style the axes
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Set static plot limits to encompass the entire tour area, centered on Sol
    view_radius_ly = 20.0
    ax.set_xlim(-view_radius_ly, view_radius_ly)
    ax.set_ylim(-view_radius_ly, view_radius_ly)
    ax.set_zlim(-view_radius_ly, view_radius_ly)

    # Plot all stars in the dataset
    star_coords = np.array([s.coords for s in all_stars.values()])
    ax.scatter(star_coords[:, 0], star_coords[:, 1], star_coords[:, 2], s=10, c='white', alpha=0.7)

    # Label the stars in the tour
    tour_star_names = set(tour)
    for name, star in all_stars.items():
        if name in tour_star_names:
            ax.text(star.coords[0], star.coords[1], star.coords[2], f' {name}', color='cyan', fontsize=9)

    # --- 4. Prepare for animation ---
    # A deque to hold the last N points of the ship's trail
    trail_deque = collections.deque(maxlen=TRAIL_LENGTH)

    # Create a list of line artists, one for each segment of the trail
    trail_lines = [ax.plot([], [], [], lw=2)[0] for _ in range(TRAIL_LENGTH - 1)]

    # A text artist to display the current simulation time
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white')

    # --- 5. Define the animation update function ---
    total_frames = int((total_sim_duration_years / SIM_YEARS_PER_SECOND) * (1000 / ANIMATION_INTERVAL_MS))

    def animate(frame_num):
        # Calculate the current simulation time in years
        current_sim_time = (frame_num / total_frames) * total_sim_duration_years

        # Find which leg of the journey we are on
        current_leg = None
        for leg in tour_schedule:
            if leg['start_time'] <= current_sim_time < leg['end_time']:
                current_leg = leg
                break

        ship_pos = None
        if current_leg:
            # We are traveling
            time_into_leg = current_sim_time - current_leg['start_time']
            ship_pos = get_ship_position_at_sol_time(
                current_leg['start_pos'],
                current_leg['end_pos'],
                current_leg['duration'],
                time_into_leg,
                acceleration_g
            )
            # Determine if we are accelerating or decelerating
            is_accel = time_into_leg < current_leg['duration'] / 2.0
            trail_deque.append((ship_pos, is_accel))

        elif trail_deque:
            # We are resting at a star, or the tour is over.
            # Hold the ship at its last known position.
            ship_pos, is_accel = trail_deque[-1]
            trail_deque.append((ship_pos, is_accel)) # Keep the trail from disappearing

        # Update the visual trail from the deque
        for i, line in enumerate(trail_lines):
            if i < len(trail_deque) - 1:
                p1, _ = trail_deque[i]
                p2, is_accel_p2 = trail_deque[i+1]

                # Set the 3D data for this line segment
                line.set_data_3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

                # Set color and alpha to create the fading effect
                alpha = (i + 1) / TRAIL_LENGTH
                color = 'lime' if is_accel_p2 else 'red'
                line.set_color(color)
                line.set_alpha(alpha)
            else:
                # Hide unused line segments
                line.set_data_3d([], [], [])

        # Update the time display
        time_text.set_text(f'Sol Time: {current_sim_time:.2f} years')

        # Return all artists that were modified
        return trail_lines + [time_text]

    # --- 6. Run the animation ---
    print("\nStarting animation... Close the plot window to exit.")
    # We use blit=True for performance, which means the animate function must
    # return an iterable of all artists that have been modified.
    ani = FuncAnimation(
        fig,
        animate,
        frames=total_frames,
        interval=ANIMATION_INTERVAL_MS,
        blit=True,
        repeat=False
    )

    plt.show()

if __name__ == "__main__":
    main()