import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def sailing_polar_curve(wind_speed, angle_degrees):
    """
    Calculates the boat speed based on wind speed and angle to the wind.
    This is a simplified model.  Real polar curves are much more complex.

    Args:
        wind_speed (float): The wind speed in knots.
        angle_degrees (float): The angle to the wind in degrees (0-180).
                             0 is directly upwind, 180 is directly downwind.

    Returns:
        float: The boat speed in knots.
    """
    angle_radians = np.radians(angle_degrees)

    # Convert angle to a range of 0 to pi
    angle_radians = np.abs(angle_radians)
    if angle_radians > np.pi:
        angle_radians = 2 * np.pi - angle_radians

    # Simplified model with some made-up parameters
    max_speed = wind_speed * 0.8  # Max boat speed is a fraction of wind speed
    optimal_angle = np.radians(60)  # Optimal angle to the wind

    # Ensure wind speed is always positive.
    wind_speed = max(0, wind_speed)
    # Avoid division by zero or negative values in the sqrt.
    if wind_speed <= 0:
        return 0.0

    # Model boat speed as a function of wind angle, with a peak at optimal_angle
    speed_factor = np.exp(-((angle_radians - optimal_angle) ** 2) / (2 * (np.pi / 6) ** 2))
    boat_speed = max_speed * speed_factor * np.sqrt(wind_speed) # Include wind speed تاثیر
    return boat_speed

# Generate data for the polar plot
wind_speed = 15  # Example wind speed of 15 knots
angles_degrees = np.linspace(0, 180, 181)  # Angles from 0 to 180 degrees
boat_speeds = [sailing_polar_curve(wind_speed, angle) for angle in angles_degrees]

# Convert to radians for plotting
angles_radians = np.radians(angles_degrees)

# Create the polar plot
plt.figure(figsize=(8, 8))  # Make the plot square
ax = plt.subplot(111, projection='polar')
ax.plot(angles_radians, boat_speeds, color='b', linewidth=2)
ax.set_title(f'Sailing Polar Curve (Wind Speed = {wind_speed} knots)')
ax.set_theta_zero_location("N")  # Set 0 degrees to North
ax.set_theta_direction(-1)  # Clockwise direction
ax.set_rlabel_position(22.5)  # Move the radius labels slightly out

# Set the angle labels
angle_labels = ['0°', '30°', '60°', '90°', '120°', '150°', '180°']
ax.set_xticks(np.radians(np.arange(0, 181, 30)))
ax.set_xticklabels(angle_labels)
ax.grid(True)
plt.show()


#  Demonstrate the function with a few examples
print("Examples:")
print(f"At 30 degrees, boat speed: {sailing_polar_curve(10, 30):.2f} knots")
print(f"At 60 degrees, boat speed: {sailing_polar_curve(10, 60):.2f} knots")
print(f"At 90 degrees, boat speed: {sailing_polar_curve(10, 90):.2f} knots")
print(f"At 120 degrees, boat speed: {sailing_polar_curve(10, 120):.2f} knots")
print(f"At 150 degrees, boat speed: {sailing_polar_curve(10, 150):.2f} knots")
print(f"At 180 degrees, boat speed: {sailing_polar_curve(10, 180):.2f} knots")

print("\nTesting with zero wind speed:")
print(f"At 60 degrees with 0 wind, boat speed: {sailing_polar_curve(0, 60):.2f} knots")

print("\nTesting with negative wind speed (should be zero):")
print(f"At 60 degrees with -5 wind, boat speed: {sailing_polar_curve(-5, 60):.2f} knots")