import time
import pathlib
import argparse
import random
import sys
import os
import multiprocessing

# To make this work, you might need to run python with the `-m` flag
# from the parent directory, e.g., `python -m Stellar_TSP.main ...`
# Or set up a proper package structure.
import stardata
import solver
import ui

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # In development, the base path is the project root
        base_path = pathlib.Path(__file__).parent.resolve()

    return os.path.join(base_path, relative_path)

def main():
    """Main execution function."""
    start_time = time.perf_counter()

    # --- Data File Selection ---
    try:
        # We assume CSVs are in a 'data' subfolder
        data_dir = resource_path('data')
        available_csvs = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    except FileNotFoundError:
        print("\nError: 'data' directory not found. This directory must exist in your project.")
        input("Press Enter to exit.")
        return

    if not available_csvs:
        print("\nError: No .csv files found in the 'data' directory.")
        input("Press Enter to exit.")
        return

    chosen_csv_name = ui.get_user_csv_choice(available_csvs)
    csv_path = os.path.join(data_dir, chosen_csv_name)

    print(f"Loading star data from '{csv_path}'...")
    all_stars = stardata.load_star_data(csv_path)
    print(f"Loaded data for {len(all_stars)} stars.")

    max_visits = len(all_stars) - 1 if len(all_stars) > 1 else 1
    num_to_visit = ui.get_number_of_stops(max_visits)

    start_star, end_star = ui.get_user_start_and_end_stars(all_stars)
    acceleration_g = ui.get_user_acceleration()
    rest_time_years = ui.get_user_rest_time()

    # Determine intermediate stops based on user's choice
    intermediate_stops = []
    if num_to_visit > 1:
        potential_intermediate = list(all_stars.keys())
        potential_intermediate.remove(start_star)
        if start_star != end_star:
            try:
                potential_intermediate.remove(end_star)
            except ValueError:
                pass # This can happen if end_star was not in the list, which is an error caught later.
        
        num_intermediate = num_to_visit - 1
        k = min(num_intermediate, len(potential_intermediate))
        intermediate_stops = random.sample(potential_intermediate, k=k)

    try:
        tour, ship_time, sol_time, leg_details = solver.solve_gr_tsp_nearest_neighbor(
            all_stars, start_star, end_star, intermediate_stops, acceleration_g, rest_time_years
        )

        ui.print_tour_summary(tour, leg_details)
        ui.print_final_analysis(ship_time, sol_time, leg_details, acceleration_g, rest_time_years)

    except (ValueError, RuntimeError) as e:
        print(f"\nAn error occurred during tour calculation: {e}")

    end_time = time.perf_counter()
    ui.print_execution_time(start_time, end_time)


if __name__ == "__main__":
    # This check is crucial for multiprocessing on Windows
    # It prevents child processes from re-executing the main script block.
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as e:
        import traceback
        print("\n--- AN UNHANDLED ERROR OCCURRED ---")
        print("The program has crashed. Please see the error details below:")
        print(traceback.format_exc())

    # This will run after a successful execution or after an error has been printed,
    # keeping the window open until the user presses Enter.
    print("\nProgram has finished. Press Enter to exit.")
    input()
