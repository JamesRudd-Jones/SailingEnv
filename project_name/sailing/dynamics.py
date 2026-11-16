"""
3DOF sailing dinghy dynamics (surge, sway, yaw) with quasi-static heel.

Pure and per boat, with the wind as an input, so that several boats are a vmap. Boat speed comes from force balance
rather than polars, and every foil stalls and handles all 360 degrees of inflow.

Conventions, see angles.py for the conversion to compass: world x east, world y north, heading anticlockwise from +x,
body x forward, body y to port, yaw rate anticlockwise. Positive rudder turns the boat to starboard.

The state array is (x, y, heading, u, v, r, boom, rudder). A positive boom angle has the boom out to starboard.
"""

from typing import Tuple

import chex
from flax import struct
import jax
import jax.numpy as jnp


NUM_STATES = 8

X, Y, HEADING, U, V, R, BOOM, RUDDER = range(NUM_STATES)


@struct.dataclass
class FoilParams:
    area: float  # m^2
    aspect_ratio: float
    stall_angle: float  # rad, centre of the transition into separated flow
    stall_width: float  # width of that transition, measured in sin(angle of attack)
    normal_coeff: float  # normal force coefficient of the fully separated foil, like a flat plate
    cd0: float  # profile drag coefficient
    oswald: float = 0.8  # span efficiency
    luff_angle: float = 0.0  # rad, angle of attack below which a soft sail collapses, zero for a rigid foil


@struct.dataclass
class BoatParams:
    """
    A Firefly: Uffa Fox's 12 ft two-person dinghy, with main and jib lumped into one sail.

    SOURCED, Wikipedia and Keel Index: length 3.66 m, waterline 3.58 m, beam 1.42 m, draught 1.30 m board down and 0.25 m
    board up, two crew, hull speed 4.6 kn. Ovington Boats: mast 6.4 m, upwind sail area 8.36 m^2, fitted hull 74 kg.
    Wikipedia gives 118 kg displacement, taken here as the rigged boat, and 6.32 m^2 of sail, not used.
    ESTIMATED, with the reasoning alongside: everything else. No measured Firefly polars were found to fit against.
    """
    # 118 kg rigged boat plus an assumed 135 kg crew is 253 kg, before added mass
    mass_x: float = 270.0  # kg, about 7% surge added mass
    mass_y: float = 410.0  # kg, board about 100 kg from rho * pi * (chord / 2)^2 * span, hull about 60 kg
    inertia_z: float = 270.0  # kg m^2, radius of gyration about a quarter of the length, plus about 30% added inertia

    water_density: float = 1000.0  # kg m^-3
    air_density: float = 1.2  # kg m^-3

    # Stall at 28 degrees because a main and jib together hold their lift further than one soft sail. cd0 is mostly
    # windage: two crew sitting out are about 1.2 m^2 at CD 1, or 0.14 of the sail area, plus hull, rig and sail profile
    sail: FoilParams = FoilParams(area=8.36, aspect_ratio=3.5, stall_angle=jnp.radians(28.0), stall_width=0.06,
                                  normal_coeff=1.6, cd0=0.20, luff_angle=jnp.radians(10.0))
    # Stall widths are deliberately not narrow. Lift that falls too steeply with angle creates fast growth in the model
    # itself, around 40 s^-1 at a width of 0.03, which no shorter timestep removes
    # Board span 1.05 m is draught less hull draught, the 0.35 m chord is assumed
    board: FoilParams = FoilParams(area=0.37, aspect_ratio=3.0, stall_angle=jnp.radians(16.0), stall_width=0.12,
                                   normal_coeff=1.1, cd0=0.012)
    # Assumed 0.8 m span by 0.16 m chord, a transom-hung blade
    rudder: FoilParams = FoilParams(area=0.13, aspect_ratio=3.0, stall_angle=jnp.radians(18.0), stall_width=0.15,
                                    normal_coeff=0.9, cd0=0.012)

    board_x: float = 0.05  # m, assumed, centre of pressure forward of the centre of gravity
    rudder_x: float = -1.9  # m, the transom is 1.83 m aft of mid-length
    mast_x: float = 0.6  # m, assumed
    boom_to_centre_of_effort: float = 0.55  # m, assumed, short of the main's own because the jib pulls it forward
    centre_of_effort_height: float = 3.0  # m, assumed for a 6.4 m mast
    righting_moment: float = 1300.0  # N m, 135 kg of crew sitting out at about 1 m, there is no trapeze

    boom_time_constant: float = 0.3  # s
    rudder_time_constant: float = 0.15  # s
    rudder_max_rate: float = 1.2  # rad s^-1

    hull_viscous_coeff: float = 6.1  # N per (m/s)^2, 0.5 * rho * 3.4 m^2 wetted * Cf 0.0036 from ITTC-57 with form factor
    hull_speed: float = 2.37  # m/s, sourced, 4.6 kn
    # With these two, wave drag is about 90 N at hull speed, and flattens to about 300 N at 4.2 m/s as the dinghy planes
    hull_wave_coeff: float = 1000.0  # N
    hull_wave_saturation: float = 10.0
    hull_reverse_factor: float = 4.0  # assumed, extra surge drag going astern, from the transom
    hull_sway_coeff: float = 450.0  # N per (m/s)^2, 0.5 * rho * 0.9 m^2 hull lateral area * CD 1
    yaw_linear_damping: float = 80.0  # N m per rad/s, assumed
    yaw_quadratic_damping: float = 160.0  # N m per (rad/s)^2, assumed


def foil_force(flow_x: chex.Array,
               flow_y: chex.Array,
               chord_angle: chex.Array,
               foil: FoilParams,
               density: float,
               ) -> Tuple[chex.Array, chex.Array]:
    """
    Force on a foil from the fluid velocity relative to it, valid for all 360 degrees of inflow.

    chord_angle is the direction of the leading edge in the same frame as the flow. Attached lift and induced drag are
    blended into the force on a separated flat plate, which acts normal to the chord. Everything is written in the sine
    and cosine of the angle of attack, so there are no angle wraps or sign changes to create force jumps.
    """
    speed_squared = flow_x * flow_x + flow_y * flow_y + 1e-6
    speed = jnp.sqrt(speed_squared)
    dir_x, dir_y = flow_x / speed, flow_y / speed

    chord_x, chord_y = jnp.cos(chord_angle), jnp.sin(chord_angle)
    sin_aoa = dir_y * chord_x - dir_x * chord_y
    cos_aoa = -(dir_x * chord_x + dir_y * chord_y)

    separated = jax.nn.sigmoid((jnp.sqrt(sin_aoa * sin_aoa + 1e-8) - jnp.sin(foil.stall_angle)) / foil.stall_width)

    lift_slope = 2.0 * jnp.pi / (1.0 + 2.0 / foil.aspect_ratio)
    sin_squared = sin_aoa * sin_aoa
    luff = sin_squared / (sin_squared + jnp.sin(foil.luff_angle) ** 2 + 1e-8)  # a soft sail cannot hold its shape
    attached_lift = lift_slope * sin_aoa * cos_aoa * luff
    attached_drag = foil.cd0 + attached_lift * attached_lift / (jnp.pi * foil.oswald * foil.aspect_ratio)

    separated_lift = foil.normal_coeff * sin_aoa * cos_aoa
    separated_drag = foil.cd0 + foil.normal_coeff * sin_aoa * sin_aoa

    lift = (1.0 - separated) * attached_lift + separated * separated_lift
    drag = (1.0 - separated) * attached_drag + separated * separated_drag

    dynamic_force = 0.5 * density * foil.area * speed_squared
    return dynamic_force * (lift * dir_y + drag * dir_x), dynamic_force * (drag * dir_y - lift * dir_x)


def hull_force(u: chex.Array, v: chex.Array, r: chex.Array, params: BoatParams) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Surge force from surge speed and sway force from sway speed, so there is no force jump when u crosses zero."""
    froude = u / params.hull_speed
    wave = params.hull_wave_coeff * froude ** 3 * jnp.abs(froude) / (1.0 + params.hull_wave_saturation * froude ** 2)
    astern = jax.nn.sigmoid(-u / 0.1)
    surge = -(1.0 + (params.hull_reverse_factor - 1.0) * astern) * (params.hull_viscous_coeff * jnp.abs(u) * u + wave)
    sway = -params.hull_sway_coeff * jnp.abs(v) * v
    yaw = -(params.yaw_linear_damping + params.yaw_quadratic_damping * jnp.abs(r)) * r
    return surge, sway, yaw


def rigid_body_acceleration(u: chex.Array,
                            v: chex.Array,
                            r: chex.Array,
                            force_x: chex.Array,
                            force_y: chex.Array,
                            moment_z: chex.Array,
                            params: BoatParams,
                            ) -> chex.Array:
    """Fossen form with added mass. The velocity terms do no work, and the last one is the destabilising Munk moment."""
    return jnp.stack(((force_x + params.mass_y * v * r) / params.mass_x,
                      (force_y - params.mass_x * u * r) / params.mass_y,
                      (moment_z - (params.mass_y - params.mass_x) * u * v) / params.inertia_z))


def boom_target(apparent_wind_angle: chex.Array, sail_command: chex.Array) -> chex.Array:
    """
    A positive command is sheet length: the boom swings freely with the wind until the sheet stops it, which gives
    luffing and gybing without any special cases. A negative command holds the boom out to windward, backing the main.
    The two meet at a command of zero.
    """
    sheet = jnp.maximum(sail_command, 0.0)
    backing = jnp.minimum(sail_command, 0.0)
    return jnp.clip(apparent_wind_angle, -sheet, sheet) + backing * jnp.tanh(apparent_wind_angle / 0.1)


def forces_and_moment(boat: chex.Array, wind_world: chex.Array, params: BoatParams) -> dict:
    """All forces and the yaw moment in the body frame, along with the quantities later needed for the wake."""
    heading, u, v, r, boom, rudder = boat[HEADING], boat[U], boat[V], boat[R], boat[BOOM], boat[RUDDER]
    cos_h, sin_h = jnp.cos(heading), jnp.sin(heading)

    wind_x = cos_h * wind_world[0] + sin_h * wind_world[1]
    wind_y = cos_h * wind_world[1] - sin_h * wind_world[0]

    # The centre of effort moves with the boom, which lies aft of the mast
    effort_x = params.mast_x - params.boom_to_centre_of_effort * jnp.cos(boom)
    effort_y = -params.boom_to_centre_of_effort * jnp.sin(boom)

    apparent_x = wind_x - (u - r * effort_y)
    apparent_y = wind_y - (v + r * effort_x)
    apparent_wind_angle = jnp.arctan2(-apparent_y, -apparent_x + 1e-6)

    sail_x, sail_y = foil_force(apparent_x, apparent_y, boom, params.sail, params.air_density)

    # Quasi-static heel: drive and side force fall smoothly once the heel moment exceeds what the sailor can hold
    heel_moment = sail_y * params.centre_of_effort_height
    depower = 1.0 / jnp.sqrt(1.0 + (heel_moment / params.righting_moment) ** 2)
    sail_x, sail_y = depower * sail_x, depower * sail_y

    board_x, board_y = foil_force(-u, -(v + r * params.board_x), jnp.zeros_like(u), params.board, params.water_density)
    rudder_x, rudder_y = foil_force(-u, -(v + r * params.rudder_x), rudder, params.rudder, params.water_density)
    hull_x, hull_y, hull_n = hull_force(u, v, r, params)

    moment_z = (effort_x * sail_y - effort_y * sail_x + params.board_x * board_y + params.rudder_x * rudder_y + hull_n)

    return {"force_x": sail_x + board_x + rudder_x + hull_x,
            "force_y": sail_y + board_y + rudder_y + hull_y,
            "moment_z": moment_z,
            "sail_x": sail_x,
            "sail_y": sail_y,
            "rudder_x": rudder_x,
            "rudder_y": rudder_y,
            "board_x": board_x,
            "board_y": board_y,
            "apparent_wind_angle": apparent_wind_angle,
            "apparent_wind_speed": jnp.sqrt(apparent_x * apparent_x + apparent_y * apparent_y + 1e-6),
            }


def boat_derivative(boat: chex.Array, action: chex.Array, wind_world: chex.Array, params: BoatParams) -> chex.Array:
    """Rate of change of the boat state. The action is (rudder command, sail command) in radians."""
    heading, u, v, r, boom, rudder = boat[HEADING], boat[U], boat[V], boat[R], boat[BOOM], boat[RUDDER]
    loads = forces_and_moment(boat, wind_world, params)

    cos_h, sin_h = jnp.cos(heading), jnp.sin(heading)
    accel = rigid_body_acceleration(u, v, r, loads["force_x"], loads["force_y"], loads["moment_z"], params)

    boom_rate = (boom_target(loads["apparent_wind_angle"], action[1]) - boom) / params.boom_time_constant
    rudder_rate = jnp.clip((action[0] - rudder) / params.rudder_time_constant,
                           -params.rudder_max_rate,
                           params.rudder_max_rate)

    return jnp.stack((cos_h * u - sin_h * v, sin_h * u + cos_h * v, r, accel[0], accel[1], accel[2], boom_rate, rudder_rate))


def ode_rhs(t: chex.Array, boat: chex.Array, control: chex.Array, params: BoatParams) -> chex.Array:
    """Signature for BifurcaGym's integrate_ode, where control is (rudder command, sail command, wind x, wind y)."""
    return boat_derivative(boat, control[:2], control[2:], params)
