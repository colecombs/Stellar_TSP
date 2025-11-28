import constants
import multiprocessing

def get_user_acceleration() -> float:
    """Prompts the user to enter the ship's acceleration."""
    while True:
        try:
            prompt = (f"Enter ship's constant acceleration in Gs "
                      f"({constants.MIN_ACCELERATION_G} to {constants.MAX_ACCELERATION_G}) [{constants.DEFAULT_ACCELERATION_G}]: ")
            accel_input = input(prompt)
            if not accel_input:
                return constants.DEFAULT_ACCELERATION_G
            
            acceleration_g = float(accel_input)
            if constants.MIN_ACCELERATION_G <= acceleration_g <= constants.MAX_ACCELERATION_G:
                return acceleration_g
            else:
                print(f"Error: Acceleration must be between {constants.MIN_ACCELERATION_G} and {constants.MAX_ACCELERATION_G} Gs.")
        except ValueError:
            print("Error: Please enter a valid number.")

def get_user_rest_time() -> float:
    """Prompts the user to enter the rest time at each star."""
    while True:
        try:
            prompt = (f"Enter rest time at each star in years "
                      f"({constants.MIN_REST_TIME_YEARS} to {constants.MAX_REST_TIME_YEARS}) [{constants.DEFAULT_REST_TIME_YEARS}]: ")
            rest_input = input(prompt)
            if not rest_input:
                return constants.DEFAULT_REST_TIME_YEARS
            
            rest_time = float(rest_input)
            if constants.MIN_REST_TIME_YEARS <= rest_time <= constants.MAX_REST_TIME_YEARS:
                return rest_time
            else:
                print(f"Error: Rest time must be between {constants.MIN_REST_TIME_YEARS} and {constants.MAX_REST_TIME_YEARS} years.")
        except ValueError:
            print("Error: Please enter a valid number.")

def get_user_csv_choice(available_files: list[str]) -> str:
    """Displays a list of CSV files and prompts the user to choose one."""
    print("\nAvailable Star Datasets:")
    for i, name in enumerate(available_files):
        print(f"  {i+1:2d}: {name}")

    while True:
        try:
            # Default to the first file in the list
            prompt = f"\nEnter the number for the dataset to use: "
            choice_input = input(prompt)
            if not choice_input:
                return available_files[0]
            choice = int(choice_input)
            if 1 <= choice <= len(available_files):
                return available_files[choice - 1]
            else:
                print(f"Error: Please enter a number between 1 and {len(available_files)}.")
        except (ValueError, IndexError):
            print("Error: Please enter a valid number.")

def get_number_of_stops(max_stops: int) -> int:
    """Prompts the user for the number of stars to visit."""
    while True:
        try:
            prompt = (f"Enter the number of stars to visit (1 - {max_stops}): ")
            num_input = input(prompt)
            if not num_input:
                return max_stops # Default to all
            
            num_stops = int(num_input)
            if 1 <= num_stops <= max_stops:
                return num_stops
            else:
                print(f"Error: Number of stops must be between 1 and {max_stops}.")
        except ValueError:
            print("Error: Please enter a valid number.")

def get_user_start_and_end_stars(all_stars: dict) -> tuple[str, str]:
    """Displays a list of stars and prompts the user to pick a start and end star."""
    # Sort stars by their distance from Sol for a more logical presentation.
    sorted_star_objects = sorted(all_stars.values(), key=lambda s: s.distance_ly)
    star_list = [s.name for s in sorted_star_objects]

    print("\nAvailable Stars (sorted by distance from Sol):")
    for i, name in enumerate(star_list):
        print(f"  {i+1:2d}: {name:<25s} ({all_stars[name].distance_ly:.2f} ly)")

    # Get start star
    while True:
        try:
            prompt = f"\nEnter the number for the STARTING star [1-{len(star_list)}]: "
            choice_input = input(prompt)
            choice = int(choice_input)
            if 1 <= choice <= len(star_list):
                start_star = star_list[choice - 1]
                break
            else:
                print(f"Error: Please enter a number between 1 and {len(star_list)}.")
        except ValueError:
            print("Error: Please enter a valid number.")

    # Get end star
    while True:
        try:
            prompt = f"Enter the number for the ENDING star [1-{len(star_list)}]: "
            choice_input = input(prompt)
            choice = int(choice_input)
            if 1 <= choice <= len(star_list):
                end_star = star_list[choice - 1]
                break
            else:
                print(f"Error: Please enter a number between 1 and {len(star_list)}.")
        except ValueError:
            print("Error: Please enter a valid number.")
            
    return start_star, end_star

def get_log_preference() -> bool:
    """Asks the user if they want to create a log file."""
    while True:
        prompt = "\nCreate a log file for this tour? (y/n): "
        choice = input(prompt).lower().strip()
        if not choice or choice == 'y':
            return True
        if choice == 'n':
            return False
        print("Error: Please enter 'y' or 'n'.")

def get_num_threads() -> int:
    """Prompts the user for the number of threads to use for calculations."""
    max_threads = multiprocessing.cpu_count()
    # Suggest half the available threads, but ensure it's at least 1.
    suggested_threads = max(1, max_threads // 2)

    while True:
        try:
            prompt = f"Enter the number of threads for calculation (1 to {max_threads}) [{suggested_threads}]: "
            choice_input = input(prompt)
            if not choice_input:
                return suggested_threads
            
            choice = int(choice_input)
            if 1 <= choice <= max_threads:
                return choice
            else:
                print(f"Error: Please enter a number between 1 and {max_threads}.")
        except ValueError:
            print("Error: Please enter a valid number.")

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
        dist_str = f"{details.get('distance_ly', 0):.2f}"
        ship_time_str = f"{details.get('ship_time_years', 0):.2f}"
        sol_time_str = f"{details.get('sol_time_years', 0):.6f}"
        max_speed_str = f"{details.get('max_speed_percent_c', 0):.6f}"
        dilation_str = f"{details.get('time_dilation_factor', 1):.2f}"
        row = (f"{leg_name:<38s} | {dist_str:>15s} | {ship_time_str:>15s} | "
               f"{sol_time_str:>15s} | {max_speed_str:>15s} | {dilation_str:>15s}")
        print(row)

def print_final_analysis(ship_time, sol_time, leg_details, acceleration_g, rest_time_years):
    """Prints the final summary of the entire tour."""
    total_travel_ship_time = sum(d.get('ship_time_years', 0) for d in leg_details)
    total_travel_sol_time = sum(d.get('sol_time_years', 0) for d in leg_details)
    average_dilation = (total_travel_sol_time / total_travel_ship_time) if total_travel_ship_time > 0 else 1.0
    weighted_avg_max_speed = (sum(d.get('max_speed_percent_c', 0) * d.get('sol_time_years', 0) for d in leg_details) / total_travel_sol_time) if total_travel_sol_time > 0 else 0.0

    print("\n--- Relativistic Travel Time Analysis (GR-Aware) ---")
    print(f"Assumptions: {acceleration_g}g accel/decel, {rest_time_years}-year stay, orbital maneuvers, n-body gravity.")
    print(f"Total subjective time on ship (travel + stops): {ship_time:.2f} years")
    print(f"Total time in Sol's frame (travel + stops):   {sol_time:.2f} years")
    print(f"Average time dilation factor (travel only):     {average_dilation:.2f}x")
    print(f"Average max speed (weighted by leg duration):   {weighted_avg_max_speed:.6f}% of c")

def print_execution_time(start_time, end_time):
    """Calculates and prints the script's total execution time."""
    duration_s = end_time - start_time
    print("\n--- Script Execution Time ---")
    if duration_s > 60:
        minutes, seconds = divmod(duration_s, 60)
        print(f"Total time: {int(minutes)} minutes, {seconds:.3f} seconds")
    else:
        print(f"Total time: {duration_s:.3f} seconds")
