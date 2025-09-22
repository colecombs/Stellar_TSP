import numpy as np
from typing import Dict

import constants
from stardata import Star, predict_star_position

def get_gravity_at_point(ship_pos_m: np.ndarray, time_years: float, all_stars: Dict[str, Star], exclude_star_name: str = None) -> np.ndarray:
    """
    Calculates the total Newtonian gravitational acceleration vector at a point in space.
    This serves as a first-order approximation for the effects of curved spacetime.
    """
    total_accel_m_s2 = np.zeros(3)
    for star in all_stars.values():
        if star.mass_sol == 0:
            continue

        if star.name == exclude_star_name:
            orbital_radius_m = star.mass_sol * constants.AU_METERS
            if np.sum((predict_star_position(star, time_years) * constants.METERS_PER_LIGHT_YEAR - ship_pos_m)**2) < orbital_radius_m**2:
                continue

        star_pos_ly = predict_star_position(star, time_years)
        star_pos_m = star_pos_ly * constants.METERS_PER_LIGHT_YEAR

        r_vec = star_pos_m - ship_pos_m
        dist_sq = np.sum(r_vec**2)
        if dist_sq == 0:
            continue
        
        dist = np.sqrt(dist_sq)
        force_dir = r_vec / dist
        
        star_mass_kg = star.mass_sol * constants.SOL_MASS_KG
        accel_magnitude = constants.G_GRAVITATIONAL_CONSTANT * star_mass_kg / dist_sq
        total_accel_m_s2 += accel_magnitude * force_dir
        
    return total_accel_m_s2

def get_star_velocity(star_data: Star, time_years: float) -> np.ndarray:
    """
    Calculates the velocity vector of a star at a given time by numerical differentiation.
    """
    # A small time step, e.g., one day, in units of years.
    dt_years = 1.0 / 365.25
    
    pos1_ly = predict_star_position(star_data, time_years)
    pos2_ly = predict_star_position(star_data, time_years + dt_years)
    
    vel_ly_per_year = (pos2_ly - pos1_ly) / dt_years
    
    return vel_ly_per_year * constants.METERS_PER_LIGHT_YEAR / constants.SECONDS_PER_YEAR

def calculate_orbital_maneuver(star: Star, interstellar_velocity_ms: float, acceleration_g: float) -> tuple[float, float]:
    """
    Calculates the delta-v and time required for an orbital maneuver.
    This covers both escape from orbit and insertion into orbit.
    """
    if star.mass_sol == 0:
        return 0.0, 0.0

    orbital_radius_m = star.mass_sol * constants.AU_METERS
    star_mass_kg = star.mass_sol * constants.SOL_MASS_KG

    v_orbit_ms = np.sqrt(constants.G_GRAVITATIONAL_CONSTANT * star_mass_kg / orbital_radius_m)
    v_escape_ms = np.sqrt(2 * constants.G_GRAVITATIONAL_CONSTANT * star_mass_kg / orbital_radius_m)
    v_post_burn_ms = np.sqrt(interstellar_velocity_ms**2 + v_escape_ms**2)
    delta_v_ms = v_post_burn_ms - v_orbit_ms
    burn_time_s = delta_v_ms / (acceleration_g * constants.G_ACCELERATION)
    
    return delta_v_ms, burn_time_s

def sr_travel_time(distance_ly: float, acceleration_g: float) -> float:
    """
    A helper function to get a quick Special Relativity-based estimate for travel time.
    Used to provide a reasonable time-span for the GR numerical integration.
    """
    d_meters = distance_ly * constants.METERS_PER_LIGHT_YEAR
    if d_meters == 0: return 0.0
    a = acceleration_g * constants.G_ACCELERATION
    c = constants.C_METERS_PER_SECOND
    term = (a * d_meters) / (2 * c**2)
    sol_time_leg_sec = 2 * (c / a) * np.sqrt((1 + term)**2 - 1)
    return sol_time_leg_sec / constants.SECONDS_PER_YEAR

def ode_system_for_scipy(t, y, all_stars, target_star, acceleration_g, t_start_years, t_mid_years, start_star_name):
    """
    Defines the system of Ordinary Differential Equations for the ship's motion.
    y = [x, y, z, vx, vy, vz, proper_time]
    """
    ship_pos_m = y[:3]
    ship_vel_ms = y[3:6]
    
    current_time_years = t_start_years + (t / constants.SECONDS_PER_YEAR)
    
    accel_gravity = get_gravity_at_point(ship_pos_m, current_time_years, all_stars, start_star_name)
    
    # The ship's engines provide a constant *proper* acceleration.
    proper_accel_mag = acceleration_g * constants.G_ACCELERATION
    
    if current_time_years < t_mid_years:
        target_pos_ly = predict_star_position(target_star, current_time_years)
        target_pos_m = target_pos_ly * constants.METERS_PER_LIGHT_YEAR
        propulsion_dir = (target_pos_m - ship_pos_m)
        norm = np.linalg.norm(propulsion_dir)
        if norm > 0:
            propulsion_dir /= norm
    else:
        norm = np.linalg.norm(ship_vel_ms)
        if norm > 0:
            propulsion_dir = -ship_vel_ms / norm
        else:
            propulsion_dir = np.zeros(3)
    
    proper_accel_engine_vec = propulsion_dir * proper_accel_mag
    
    # Per the principle of equivalence, we sum the "felt" accelerations (engine + gravity)
    # before applying relativistic corrections.
    total_proper_accel_vec = proper_accel_engine_vec + accel_gravity
    
    # Now, convert the *total* proper acceleration into coordinate acceleration.
    v_sq = np.sum(ship_vel_ms**2)
    c_sq = constants.C_METERS_PER_SECOND**2

    if v_sq >= c_sq:
        # At or beyond c (due to numerical error), all acceleration is ineffective.
        total_accel = np.zeros(3)
    elif v_sq > 1e-6: # If moving at a non-trivial speed
        gamma = 1.0 / np.sqrt(1.0 - v_sq / c_sq)
        v_hat = ship_vel_ms / np.sqrt(v_sq)
        
        a_parallel_proper = np.dot(total_proper_accel_vec, v_hat) * v_hat
        a_perp_proper = total_proper_accel_vec - a_parallel_proper
        
        total_accel = a_perp_proper / gamma + a_parallel_proper / (gamma**3)
    else: # If ship is at rest or moving very slowly, gamma ~ 1.
        total_accel = total_proper_accel_vec
    
    # Proper time integration (dτ/dt)
    v_sq = np.sum(ship_vel_ms**2)
    lorentz_term = 1 - v_sq / constants.C_METERS_PER_SECOND**2
    if lorentz_term > 0:
        d_proper_time_dt = np.sqrt(lorentz_term)
    else:
        d_proper_time_dt = 0.0
    
    return [
        ship_vel_ms[0], ship_vel_ms[1], ship_vel_ms[2],
        total_accel[0], total_accel[1], total_accel[2],
        d_proper_time_dt
    ]
