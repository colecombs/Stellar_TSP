import os
import pathlib
from astroquery.gaia import Gaia

def fetch_and_save_nearby_stars():
    """
    Queries the Gaia DR3 Catalogue of Nearby Stars (GCNS) for stars
    within approximately 100 light-years and saves the data to a CSV file.
    """
    # ADQL query to get the nearest stars with all necessary data
    # Parallax > 32.8 mas corresponds to < 30.48 parsecs (~100 light-years)
    # This gives a good buffer around the nearest 1000 stars.
    adql_query = """
    SELECT
      source_id,
      plx AS parallax,
      ra,
      dec,
      pmra AS ra_motion,
      pmdec AS dec_motion,
      radial_velocity,
      ruwe,
      phot_g_mean_mag,
      grvs_mag,
      mass_val AS mass
    FROM
      gaiadr3.gcns
    WHERE
      plx > 32.8
      AND mass_val IS NOT NULL
      AND mass_val > 0
    ORDER BY
      plx DESC
    """

    print("Connecting to Gaia Archive and running query...")
    try:
        job = Gaia.launch_job_async(adql_query)
        results = job.get_results()
        print(f"Successfully retrieved {len(results)} stars from the Gaia archive.")
    except Exception as e:
        print(f"An error occurred while querying Gaia: {e}")
        return

    # Define the output path
    # Assumes a 'data' folder exists at the project root
    data_dir = pathlib.Path(__file__).parent / 'data'
    os.makedirs(data_dir, exist_ok=True)
    output_path = data_dir / 'gaia_nearby_stars.csv'

    # Rename columns to match what stardata.py expects
    results.rename_column('source_id', 'star_name')
    results.rename_column('parallax', 'distance_ly') # Placeholder, will be calculated
    results.rename_column('ra', 'ra_hms') # Placeholder, will be calculated
    results.rename_column('dec', 'dec_deg')
    results.rename_column('ra_motion', 'RA_Motion')
    results.rename_column('dec_motion', 'Dec_Motion')
    results.rename_column('radial_velocity', 'Rad_v(km/s)')

    # --- Data Cleaning and Formatting ---
    # Your stardata.py expects distance in light-years and RA in hms format.
    # We need to convert the raw query results.
    print(f"Formatting data and saving to '{output_path}'...")
    
    # Fill any missing motion or velocity values with 0, as your code expects numbers.
    for col in ['RA_Motion', 'Dec_Motion', 'Rad_v(km/s)']:
        if col in results.colnames:
            results[col].fill_value = 0
            results = results.filled()

    # Convert parallax (mas) to distance (ly)
    # distance_ly = (1 / (parallax_mas / 1000)) * 3.26156
    results['distance_ly'] = (1.0 / (results['parallax'] / 1000.0)) * 3.26156

    # Convert RA (degrees) to HMS string format for stardata.py
    ra_hms_list = []
    for ra_deg in results['ra']:
        hours = int(ra_deg / 15.0)
        minutes = int(((ra_deg / 15.0) - hours) * 60)
        seconds = ((((ra_deg / 15.0) - hours) * 60) - minutes) * 60
        ra_hms_list.append(f"{hours}:{minutes}:{seconds:.4f}")
    results['ra_hms'] = ra_hms_list

    # Select and reorder columns to match the expected CSV format
    final_columns = [
        'star_name', 'distance_ly', 'ra_hms', 'dec_deg',
        'RA_Motion', 'Dec_Motion', 'Rad_v(km/s)', 'mass'
    ]
    
    # Filter to only the columns your program needs
    final_results = results[final_columns]

    # Write to CSV
    final_results.write(output_path, format='csv', overwrite=True)
    print("Done.")


if __name__ == "__main__":
    fetch_and_save_nearby_stars()

