import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import datetime
import pathlib

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

import constants
import physics
from stardata import Star, predict_star_position

def solve_leg_with_gravity(args_tuple):
    """
    Worker function to solve a single leg of the journey using GR approximations.
    Designed to be called by the multiprocessing pool.
    """
    start_star, end_star, all_stars, departure_sol_time, acceleration_g = args_tuple
    
    start_pos_ly = predict_star_position(start_star, departure_sol_time)
    
    end_pos_guess_ly = predict_star_position(end_star, departure_sol_time)
    dist_guess_ly = np.linalg.norm(end_pos_guess_ly - start_pos_ly)
    time_guess_years = physics.sr_travel_time(dist_guess_ly, acceleration_g)
    
    if time_guess_years < 1e-3:
        return {'sol_time_years': 0, 'ship_time_years': 0, 'start_star': start_star.name, 'end_star': end_star.name}

    t_mid_years = departure_sol_time + (time_guess_years / 2.0)

    # --- Event function to stop integration at closest approach ---
    def closest_approach_event(t, y, all_stars, target_star, acceleration_g, t_start_years, t_mid_years, start_star_name):
        ship_pos_m = y[:3]
        ship_vel_ms = y[3:6]
        current_time_years = t_start_years + (t / constants.SECONDS_PER_YEAR)
        
        target_pos_m = predict_star_position(target_star, current_time_years) * constants.METERS_PER_LIGHT_YEAR
        target_vel_ms = physics.get_star_velocity(target_star, current_time_years)
        
        relative_pos = ship_pos_m - target_pos_m
        relative_vel = ship_vel_ms - target_vel_ms
        
        # The event triggers when the dot product of relative position and velocity is zero.
        return np.dot(relative_pos, relative_vel)

    closest_approach_event.terminal = True  # Stop integration when event occurs
    closest_approach_event.direction = 1    # Trigger only when dot product goes from negative to positive

    v_mag_guess = (dist_guess_ly * constants.METERS_PER_LIGHT_YEAR) / (time_guess_years * constants.SECONDS_PER_YEAR)

    def residual(initial_velocity_vector):
        """
        This is the function that scipy.optimize.root will try to drive to zero.
        """
        vx, vy, vz = initial_velocity_vector
        
        y0 = np.array([
            start_pos_ly[0] * constants.METERS_PER_LIGHT_YEAR,
            start_pos_ly[1] * constants.METERS_PER_LIGHT_YEAR,
            start_pos_ly[2] * constants.METERS_PER_LIGHT_YEAR,
            vx, vy, vz, 0.0
        ])
        
        t_span_s = [0, time_guess_years * constants.SECONDS_PER_YEAR * 2.5]
        
        sol = solve_ivp(
            physics.ode_system_for_scipy,
            t_span_s,
            y0,
            args=(all_stars, end_star, acceleration_g, departure_sol_time, t_mid_years, start_star.name),
            dense_output=True,
            method='DOP853',
            events=closest_approach_event
        )
        
        # If the event triggered, the point of closest approach is the result.
        if sol.t_events[0].size > 0:
            final_ship_pos_m = sol.y_events[0][0][:3]
            final_time_s = sol.t_events[0][0]
        else:
            # If no closest approach (e.g., a bad shot), use the endpoint.
            final_ship_pos_m = sol.y[:3, -1]
            final_time_s = sol.t[-1]

        final_time_years = departure_sol_time + (final_time_s / constants.SECONDS_PER_YEAR)
        final_target_pos_m = predict_star_position(end_star, final_time_years) * constants.METERS_PER_LIGHT_YEAR
        
        miss_vector = final_ship_pos_m - final_target_pos_m
        return miss_vector

    initial_dir = end_pos_guess_ly - start_pos_ly
    norm = np.linalg.norm(initial_dir)
    if norm > 0:
        initial_dir /= norm
    initial_velocity_guess = initial_dir * v_mag_guess
    
    # Track the number of solver function evaluations
    total_nfev = 0

    # Tiered solver approach: Try for high precision first, then fall back to a looser tolerance.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # --- Tier 1: Quick & Precise ---
        # Try for a 100,000 km tolerance with a small number of iterations.
        options_tight = {'maxfev': 100}
        tol_tight_m = 100000.0 * 1000.0
        solution = root(
            residual, initial_velocity_guess, method='hybr', tol=tol_tight_m, options=options_tight
        )
        total_nfev += solution.nfev

        # --- Tier 2: Robust Fallback ---
        # If the first failed, use a 1,000,000 km tolerance and more iterations.
        if not solution.success:
            options_medium = {'maxfev': 1000}
            tol_loose_m = 1000000.0 * 1000.0
            solution = root(
                residual, initial_velocity_guess, method='hybr', tol=tol_loose_m, options=options_medium
            )
            total_nfev += solution.nfev

        # --- Tier 3: Desperation Mode ---
        # For very difficult legs, use a very loose tolerance to find any solution.
        if not solution.success:
            options_desperate = {'maxfev': 4000}
            tol_desperate_m = 10000000.0 * 1000.0 # 10 million km
            solution = root(
                residual, initial_velocity_guess, method='hybr', tol=tol_desperate_m, options=options_desperate
            )
            total_nfev += solution.nfev

    if not solution.success:
        return {'sol_time_years': float('inf'), 'ship_time_years': float('inf'), 'start_star': start_star.name, 'end_star': end_star.name, 'solver_attempts': total_nfev}

    final_velocity = solution.x

    y0 = np.array([
        start_pos_ly[0] * constants.METERS_PER_LIGHT_YEAR, start_pos_ly[1] * constants.METERS_PER_LIGHT_YEAR, start_pos_ly[2] * constants.METERS_PER_LIGHT_YEAR,
        final_velocity[0], final_velocity[1], final_velocity[2], 0.0
    ])
    t_span_s = [0, time_guess_years * constants.SECONDS_PER_YEAR * 2.5]
    
    sol = solve_ivp(
        physics.ode_system_for_scipy, t_span_s, y0,
        args=(all_stars, end_star, acceleration_g, departure_sol_time, t_mid_years, start_star.name),
        dense_output=True, method='DOP853',
        events=closest_approach_event
    )

    # On the final, successful run, get the precise times from the event.
    if sol.t_events[0].size > 0:
        sol_time_s = sol.t_events[0][0]
        final_state = sol.y_events[0][0]
        ship_time_s = final_state[6]
        final_ship_vel_ms = final_state[3:6]
    else:
        # Fallback, though this shouldn't be reached if the solver was
        # successful. If no event was triggered, the final state is simply
        # the end state of the numerical integration.

        sol_time_s = sol.t[-1]
        ship_time_s = sol.y[6, -1]
        final_ship_vel_ms = sol.y[3:6, -1]
    
    initial_interstellar_vel = np.linalg.norm(y0[3:6])
    final_interstellar_vel = np.linalg.norm(final_ship_vel_ms)
    
    _, departure_burn_time_s = physics.calculate_orbital_maneuver(start_star, initial_interstellar_vel, acceleration_g)
    _, arrival_burn_time_s = physics.calculate_orbital_maneuver(end_star, final_interstellar_vel, acceleration_g)
    
    total_burn_time_s = departure_burn_time_s + arrival_burn_time_s

    total_sol_time_s = sol_time_s + total_burn_time_s
    total_ship_time_s = ship_time_s + total_burn_time_s

    # --- Post-integration velocity clamping ---
    # The numerical integrator can sometimes slightly exceed c due to the nature of its
    # fixed-step calculations. We clamp the velocity at each step to c to correct for this artifact.
    velocities_ms = sol.y[3:6, :]
    speeds_ms = np.linalg.norm(velocities_ms, axis=0)
    
    overspeed_indices = np.where(speeds_ms >= constants.C_METERS_PER_SECOND)[0]
    
    if overspeed_indices.size > 0:
        # Create a copy of the solution's y-array to modify, as sol.y is read-only.
        corrected_y = np.copy(sol.y)
        for i in overspeed_indices:
            # Clamp the speed to c while preserving the direction of the velocity vector
            corrected_y[3:6, i] = (velocities_ms[:, i] / speeds_ms[i]) * (constants.C_METERS_PER_SECOND * 0.99999999)
        sol.y = corrected_y # The corrected data is now used for max speed calculation

    # Recalculate final positions for distance measurement
    final_target_pos_m = predict_star_position(end_star, departure_sol_time + total_sol_time_s / constants.SECONDS_PER_YEAR) * constants.METERS_PER_LIGHT_YEAR
    distance_m = np.linalg.norm(final_target_pos_m - (y0[:3]))

    max_speed_ms = np.max(np.linalg.norm(sol.y[3:6, :], axis=0))

    return {
        'sol_time_years': total_sol_time_s / constants.SECONDS_PER_YEAR,
        'ship_time_years': total_ship_time_s / constants.SECONDS_PER_YEAR,
        'distance_ly': distance_m / constants.METERS_PER_LIGHT_YEAR,
        'max_speed_percent_c': (max_speed_ms / constants.C_METERS_PER_SECOND) * 100,
        'time_dilation_factor': total_sol_time_s / total_ship_time_s if total_ship_time_s > 0 else 1,
        'start_star': start_star.name,
        'end_star': end_star.name,
        'solver_attempts': total_nfev
    }


def relativistic_velocity_addition(v1, v2):
    """
    Relativistically adds velocity v2 to v1. Assumes v1 and v2 are numpy arrays.
    """
    c = constants.C_METERS_PER_SECOND
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    if v1_mag >= c or v2_mag >= c:
        return v1  # One velocity is already c, so the result will be c

    # Simplified formula assuming v1 and v2 are parallel (or anti-parallel)
    return (v1 + v2) / (1 + np.dot(v1, v2) / c**2)

def solve_gr_tsp_nearest_neighbor(stars, start_star: str, end_star: str, intermediate_stops: list[str], acceleration_g: float, rest_time_years: float):
    """
    Solves the TSP using the GR-aware Nearest Neighbor heuristic.
    """
    if start_star not in stars or end_star not in stars:
        raise ValueError("Start or end star not found in the dataset.")

    # Set of stars to visit between start and end
    intermediate_stars = set(intermediate_stops)

    # Validate that all chosen intermediate stars exist in the main dataset
    if not intermediate_stars.issubset(stars.keys()):
        invalid_stars = intermediate_stars - set(stars.keys())
        raise ValueError(f"Intermediate stars not found in dataset: {invalid_stars}")

    tour = [start_star]
    current_star_name = start_star
    total_ship_time = 0.0
    total_sol_time = 0.0
    cumulative_sol_time = 0.0
    leg_details_list = []

    # --- Setup Log File ---
    script_dir = pathlib.Path(__file__).parent
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True) # Create the /logs directory if it doesn't exist

    log_filename = f"gr_tsp_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_filepath = logs_dir / log_filename
    print(f"\nLogging detailed output to: {log_filepath}")

    with open(log_filepath, 'w', encoding='utf-8') as log_file:
        start_time = datetime.datetime.now()
        try:
            log_file.write(f"Stellar TSP Log - {datetime.datetime.now().isoformat()}\n")
            log_file.write(f"Acceleration: {acceleration_g} G\n")
            log_file.write(f"Start Star: {start_star}\n")
            log_file.write(f"End Star: {end_star}\n")
            log_file.write("="*50 + "\n\n")
            log_file.flush()
            
            log_file.write("\n--- Leg-by-Leg Analysis ---\n")
            log_file.write(f"{'Leg':<38s} | {'Distance (ly)':>15s} | {'Ship Time (y)':>15s} | "
                           f"{'Sol Time (y)':>15s} | {'Max Speed (%c)':>15s}\n")
            log_file.write("-" * 120 + "\n")
            log_file.flush()

            def log_leg_details(start, end, result):
                log_file.write(f"{f'{start} -> {end}':<38s} | {result['distance_ly']:>15.2f} | {result['ship_time_years']:>15.2f} | "
                               f"{result['sol_time_years']:>15.6f} | {result['max_speed_percent_c']:>15.6f}\n")
                log_file.flush()





            print("Starting GR Tour Calculation. This will take some time...")
            print(f"Using {constants.NUM_THREADS} threads. Press CTRL+C to interrupt and see partial results.")

            with ProcessPoolExecutor(max_workers=constants.NUM_THREADS) as executor:
                while intermediate_stars:
                    print(f"\nCalculating legs from: {current_star_name}. Remaining intermediate: {len(intermediate_stars)}")
                    log_file.write(f"--- Calculating legs from: {current_star_name} (Time: {cumulative_sol_time:.2f} y) ---\n")
                    
                    tasks = {}
                    for next_star_name in intermediate_stars:
                        args = (stars[current_star_name], stars[next_star_name], stars, cumulative_sol_time, acceleration_g)
                        submit_time = datetime.datetime.now()
                        future = executor.submit(solve_leg_with_gravity, args)
                        tasks[future] = (stars[current_star_name].name, next_star_name)
                        log_file.write(f"  SUBMITTED: {tasks[future][0]} -> {tasks[future][1]}\n")
                    log_file.flush()

                    results = []
                    completed_count = 0
                    total_tasks = len(tasks)
                    for future in as_completed(tasks):
                        start, end = tasks[future]
                        completed_count += 1
                        try:
                            result = future.result(timeout=constants.LEG_CALCULATION_TIMEOUT_S)
                            results.append(result)
                            print(f"  ({completed_count}/{total_tasks}) > Completed leg: {start} -> {end} (Sol Time: {result['sol_time_years']:.2f} y)")
                            completion_time = datetime.datetime.now()                            
                            log_leg_details(start,end,result)
                            elapsed_time = completion_time - start_time
                            log_file.write(f"  COMPLETED: {start} -> {end}. Sol Time: {result['sol_time_years']:.2f} y. Attempts: {result.get('solver_attempts', 'N/A')}.  Total running time: {elapsed_time}\n")
                        except TimeoutError:
                            print(f"  ({completed_count}/{total_tasks}) > A leg calculation timed out: {start} -> {end}")
                            log_file.write(f"  TIMEOUT: {start} -> {end} after {constants.LEG_CALCULATION_TIMEOUT_S}s\n")
                        except Exception as e:
                            print(f"  ({completed_count}/{total_tasks}) > A leg calculation failed: {start} -> {end}")
                            log_file.write(f"  FAILED: {start} -> {end}. Error: {e}\n")
                        finally:
                            log_file.flush()

                    if not results:
                        raise RuntimeError("All leg calculations failed. Cannot continue tour.")
                        
                    best_leg_details = min(results, key=lambda r: r['sol_time_years'])

                    if best_leg_details['sol_time_years'] == float('inf'):
                        raise RuntimeError(f"Could not find a valid path from {current_star_name} to any remaining star. Tour aborted.")

                    best_next_star = best_leg_details['end_star']
                    
                    leg_details_list.append(best_leg_details)
                    total_ship_time += best_leg_details['ship_time_years']
                    total_sol_time += best_leg_details['sol_time_years']
                    cumulative_sol_time += best_leg_details['sol_time_years']

                    total_ship_time += rest_time_years
                    total_sol_time += rest_time_years
                    cumulative_sol_time += rest_time_years

                    current_star_name = best_next_star
                    tour.append(current_star_name)
                    intermediate_stars.remove(current_star_name)
                    log_file.write(f"\n>>> CHOSEN LEG: {best_leg_details['start_star']} -> {best_next_star}\n\n")
                    log_file.flush()

                # --- Final leg to the destination star ---
                print(f"\nCalculating final leg from: {current_star_name} -> {end_star}")
                log_file.write(f"--- Calculating final leg from: {current_star_name} -> {end_star} ---\n")
                log_file.flush()
                args = (stars[current_star_name], stars[end_star], stars, cumulative_sol_time, acceleration_g)
                final_leg_future = executor.submit(solve_leg_with_gravity, args)
                try:
                    final_leg_details = final_leg_future.result(timeout=constants.LEG_CALCULATION_TIMEOUT_S)
                    print(f"  > Completed leg: {final_leg_details['start_star']} -> {final_leg_details['end_star']} (Sol Time: {final_leg_details['sol_time_years']:.2f} y)")
                    log_file.write(f"  COMPLETED: {final_leg_details['start_star']} -> {final_leg_details['end_star']}. Sol Time: {final_leg_details['sol_time_years']:.2f} y. Attempts: {final_leg_details.get('solver_attempts', 'N/A')}\n")
                except TimeoutError:
                    print(f"  > Final leg calculation timed out after {constants.LEG_CALCULATION_TIMEOUT_S} seconds.")
                    log_file.write(f"  TIMEOUT: {current_star_name} -> {end_star} after {constants.LEG_CALCULATION_TIMEOUT_S}s\n")
                    final_leg_details = {'start_star': current_star_name, 'end_star': end_star, 'sol_time_years': float('inf'), 'ship_time_years': float('inf')}
                except Exception as e:
                    print(f"  > Final leg calculation failed: {e}")
                    log_file.write(f"  FAILED: {current_star_name} -> {end_star}. Error: {e}\n")
                    final_leg_details = {'start_star': current_star_name, 'end_star': end_star, 'sol_time_years': float('inf'), 'ship_time_years': float('inf')}
                finally:
                    log_file.flush()

                leg_details_list.append(final_leg_details)
                total_ship_time += final_leg_details.get('ship_time_years', 0)
                total_sol_time += final_leg_details.get('sol_time_years', 0)
                tour.append(end_star)

        except KeyboardInterrupt:
            print("\n\nTour calculation interrupted by user. Showing results for completed legs.")
            log_file.write("\n" + "="*50 + "\n")
            log_file.write("Tour calculation interrupted by user.\n")

    return tour, total_ship_time, total_sol_time, leg_details_list
