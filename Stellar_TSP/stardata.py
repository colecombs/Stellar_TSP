import csv
import math
import pathlib
from typing import Dict, Optional
from dataclasses import dataclass

import numpy as np

import constants

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

    rad_v_ly_per_year = star_data.rad_v_km_s * (constants.SECONDS_PER_YEAR / constants.KM_PER_LY)
    new_dist_ly = star_data.distance_ly + (rad_v_ly_per_year * time_years)

    ra_motion_deg_yr = star_data.ra_motion_mas_yr / (1000 * constants.ARCSECONDS_PER_DEGREE)
    dec_motion_deg_yr = star_data.dec_motion_mas_yr / (1000 * constants.ARCSECONDS_PER_DEGREE)

    new_ra_deg = star_data.ra_deg + (ra_motion_deg_yr * time_years)
    new_dec_deg = star_data.dec_deg + (dec_motion_deg_yr * time_years)

    new_ra_rad = math.radians(new_ra_deg)
    new_dec_rad = math.radians(new_dec_deg)

    x = new_dist_ly * math.cos(new_dec_rad) * math.cos(new_ra_rad)
    y = new_dist_ly * math.cos(new_dec_rad) * math.sin(new_ra_rad)
    z = new_dist_ly * math.sin(new_dec_rad)

    return np.array([x, y, z])
