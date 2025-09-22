import argparse
import pathlib
import collections
import math

import numpy as np
import plotly.graph_objects as go

# This script requires the main solver script to be in the same directory.
# It also requires numpy and plotly:
# pip install numpy plotly
try:
    import solve_star_tsp_3 as star_tsp
except ImportError:
    print("Error: Could not import 'solve_star_tsp_3.py'.")
    print("Please ensure that 'solve_star_tsp_3.py' is in the same directory as this script.")
    exit(1)

# --- Animation Constants ---
TRAIL_LENGTH = 150  # Number of points in the ship's trail
ANIMATION_FRAME_DURATION_MS = 20  # Milliseconds between frames (50 FPS)
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
    (This function is identical to the one in the Matplotlib visualizer)
    """
    direction_vector = end_pos - start_pos
    total_distance_ly = np.linalg.norm(direction_vector)

    if total_distance_ly == 0 or total_sol_time_leg == 0:
        return start_pos

    unit_direction = direction_vector / total_distance_ly

    a_ms2 = acceleration_g * star_tsp.G_ACCELERATION
    a_ly_per_year_sq = a_ms2 * (star_tsp.SECONDS_PER_YEAR**2) / star_tsp.METERS_PER_LIGHT_YEAR
    c_ly_per_year = 1.0

    def dist_from_rest(t_years: float) -> float:
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

def main():
    """Main function to set up and run the visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize a relativistic star tour using Plotly."
    )
    parser.add_argument("csv_file", help="Path to the CSV file containing star data.")
    parser.add_argument(
        "--start_star", default="Sol", help="The name of the star to start the tour from (default: Sol)."
    )
    parser.add_argument(
        "--acceleration", type=float, default=1.0, help="Ship's constant acceleration in Gs (range: 0.5 to 10.0)."
    )
    args = parser.parse_args()

    # --- 1. Load data and solve tour ---
    csv_path = pathlib.Path(args.csv_file)
    if not csv_path.is_absolute():
        script_dir = pathlib.Path(__file__).parent.resolve()
        csv_path = script_dir / csv_path

    if not (0.5 <= args.acceleration <= 10.0):
        parser.error("Acceleration must be between 0.5 and 10.0 Gs.")

    print(f"Loading star data from '{csv_path}'...")
    all_stars = star_tsp.load_star_data(csv_path)
    print(f"Loaded data for {len(all_stars)} stars.")

    print(f"\nCalculating tour from {args.start_star}...")
    tour, _, _, leg_details = star_tsp.solve_tdtsp_nearest_neighbor(all_stars, args.start_star, args.acceleration)
    print("Tour calculation complete. Preparing visualization...")

    # --- 2. Pre-calculate the full tour schedule ---
    tour_schedule = []
    cumulative_sol_time = 0.0
    for i, leg in enumerate(leg_details):
        start_star_name, end_star_name = tour[i], tour[i+1]
        start_pos = np.array(star_tsp.predict_star_position(all_stars[start_star_name], cumulative_sol_time))
        arrival_sol_time = cumulative_sol_time + leg['sol_time_years']
        end_pos = np.array(star_tsp.predict_star_position(all_stars[end_star_name], arrival_sol_time))

        tour_schedule.append({
            'start_time': cumulative_sol_time,
            'end_time': arrival_sol_time,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'duration': leg['sol_time_years'],
        })
        cumulative_sol_time = arrival_sol_time
        if i < len(leg_details) - 1:
            cumulative_sol_time += 1.0

    total_sim_duration_years = cumulative_sol_time

    # --- 3. Pre-calculate all animation frames for Plotly ---
    print("Pre-calculating animation frames...")
    frames = []
    trail_deque = collections.deque(maxlen=TRAIL_LENGTH)
    total_frames = int((total_sim_duration_years / SIM_YEARS_PER_SECOND) * (1000 / ANIMATION_FRAME_DURATION_MS))

    for frame_num in range(total_frames):
        current_sim_time = (frame_num / total_frames) * total_sim_duration_years

        current_leg = next((leg for leg in tour_schedule if leg['start_time'] <= current_sim_time < leg['end_time']), None)

        ship_pos = None
        if current_leg:
            time_into_leg = current_sim_time - current_leg['start_time']
            ship_pos = get_ship_position_at_sol_time(
                current_leg['start_pos'], current_leg['end_pos'], current_leg['duration'], time_into_leg, args.acceleration
            )
            is_accel = time_into_leg < current_leg['duration'] / 2.0
            trail_deque.append((ship_pos, is_accel))
        elif trail_deque:
            ship_pos, is_accel = trail_deque[-1]
            trail_deque.append((ship_pos, is_accel))

        # Separate trail points into accel/decel lists for coloring
        accel_x, accel_y, accel_z, accel_color = [], [], [], []
        decel_x, decel_y, decel_z, decel_color = [], [], [], []

        for i in range(len(trail_deque) - 1):
            p1, _ = trail_deque[i]
            p2, is_accel_p2 = trail_deque[i+1]
            
            # Color value for fading effect
            color_val = i / TRAIL_LENGTH

            if is_accel_p2:
                accel_x.extend([p1[0], p2[0], None])
                accel_y.extend([p1[1], p2[1], None])
                accel_z.extend([p1[2], p2[2], None])
                accel_color.extend([color_val, color_val, 0]) # Use 0 for the break, not None
            else:
                decel_x.extend([p1[0], p2[0], None])
                decel_y.extend([p1[1], p2[1], None])
                decel_z.extend([p1[2], p2[2], None])
                decel_color.extend([color_val, color_val, 0]) # Use 0 for the break, not None

        frame_name = f"frame_{frame_num}"
        frames.append(go.Frame(
            name=frame_name,
            data=[
                go.Scatter3d(x=accel_x, y=accel_y, z=accel_z, line={'color': accel_color}),
                go.Scatter3d(x=decel_x, y=decel_y, z=decel_z, line={'color': decel_color})
            ],
            layout=go.Layout(
                annotations=[go.layout.Annotation(
                    text=f'Sol Time: {current_sim_time:.2f} years',
                    align='left', showarrow=False, xref='paper', yref='paper',
                    x=0.05, y=0.95, font={'color': 'white', 'size': 14}
                )]
            )
        ))

    # --- 4. Set up the Plotly Figure ---
    fig = go.Figure()

    # Add initial (empty) traces for the trails that frames will update
    # The colorscale provides the fading effect from transparent to opaque color
    fig.add_trace(go.Scatter3d(
        mode='lines', name='Acceleration',
        line=dict(width=4, colorscale=[[0, 'rgba(0, 255, 0, 0)'], [1, 'lime']]),
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        mode='lines', name='Deceleration',
        line=dict(width=4, colorscale=[[0, 'rgba(255, 0, 0, 0)'], [1, 'red']]),
        showlegend=False
    ))

    # Add trace for all stars
    star_coords = np.array([s.coords for s in all_stars.values()])
    fig.add_trace(go.Scatter3d(
        x=star_coords[:, 0], y=star_coords[:, 1], z=star_coords[:, 2],
        mode='markers',
        marker=dict(size=3, color='white', opacity=0.7),
        text=[s.name for s in all_stars.values()],
        hoverinfo='text',
        name='Stars'
    ))

    # Add labels for tour stars
    tour_star_names = set(tour)
    tour_star_coords = np.array([s.coords for name, s in all_stars.items() if name in tour_star_names])
    tour_star_labels = [name for name in all_stars if name in tour_star_names]
    fig.add_trace(go.Scatter3d(
        x=tour_star_coords[:, 0], y=tour_star_coords[:, 1], z=tour_star_coords[:, 2],
        mode='text',
        text=tour_star_labels,
        textfont=dict(color='cyan', size=10),
        textposition='top center',
        hoverinfo='none',
        name='Tour Stars'
    ))

    # --- 5. Configure Layout and Animation Controls ---
    view_radius_ly = 20.0
    fig.update_layout(
        title=dict(text="Relativistic Star Tour", x=0.5, font=dict(color='white')),
        scene=dict(
            xaxis=dict(title='X (ly)', range=[-view_radius_ly, view_radius_ly], backgroundcolor="black", color='white', gridcolor='gray'),
            yaxis=dict(title='Y (ly)', range=[-view_radius_ly, view_radius_ly], backgroundcolor="black", color='white', gridcolor='gray'),
            zaxis=dict(title='Z (ly)', range=[-view_radius_ly, view_radius_ly], backgroundcolor="black", color='white', gridcolor='gray'),
            aspectmode='cube'
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        showlegend=False,
        # Initial time annotation
        annotations=[go.layout.Annotation(
            text='Sol Time: 0.00 years',
            align='left', showarrow=False, xref='paper', yref='paper',
            x=0.05, y=0.95, font={'color': 'white', 'size': 14}
        )],
        # Animation controls
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=ANIMATION_FRAME_DURATION_MS, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=0)
                )]
            ), dict(
                label="Pause",
                method="animate",
                args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    mode="immediate",
                    transition=dict(duration=0)
                )]
            )]
        )],
        sliders=[dict(
            steps=[dict(
                method='animate',
                args=[[f'frame_{k}'], dict(
                    mode='immediate',
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )],
                label=f'{((k / total_frames) * total_sim_duration_years):.1f}y'
            ) for k in range(0, total_frames, int(total_frames/50))], # Limit slider steps for performance
            active=0,
            transition=dict(duration=0)
        )]
    )

    fig.frames = frames

    # --- 6. Show the figure ---
    print("\nStarting visualization... A browser window should open.")
    fig.show(renderer="browser")

if __name__ == "__main__":
    main()