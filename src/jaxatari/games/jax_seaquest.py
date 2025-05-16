import os
from functools import partial
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import chex
import pygame
import jaxatari.rendering.atraJaxis as aj
import numpy as np
from gymnax.environments import spaces

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

# TODO: surface submarine at 6 divers collected + difficulty 1
# Game Constants
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

WIDTH = 160
HEIGHT = 210
SCALING_FACTOR = 3

# Colors
BACKGROUND_COLOR = (0, 0, 139)  # Dark blue for water
PLAYER_COLOR = (187, 187, 53)  # Yellow for player sub
DIVER_COLOR = (66, 72, 200)  # Pink for divers
SHARK_DIFFICULTY_COLORS = jnp.array(
    [
        [92, 186, 92],  # Level 0: Base green
        [213, 130, 74],  # Level 1: Orange (adjusted from original ROM)
        [
            170,
            92,
            170,
        ],  # Level 2: Purple (adjusted from original ROM COLOR_KILLER_SHARK_02)
        [213, 92, 130],  # Level 3: Pink (adjusted from original ROM)
        [186, 92, 92],  # Level 4: Red (adjusted from original ROM)
    ]
)
ENEMY_SUB_COLOR = (170, 170, 170)  # Gray for enemy subs
OXYGEN_BAR_COLOR = (214, 214, 214, 255)  # White for oxygen
SCORE_COLOR = (210, 210, 64)  # Score color
OXYGEN_TEXT_COLOR = (0, 0, 0)  # Black for oxygen text

# Object sizes and initial positions from RAM state
PLAYER_SIZE = (16, 11)  # Width, Height
DIVER_SIZE = (8, 11)
SHARK_SIZE = (8, 7)
ENEMY_SUB_SIZE = (8, 11)
MISSILE_SIZE = (8, 1)

PLAYER_START_X = 76
PLAYER_START_Y = 46

X_BORDERS = (0, 160)
PLAYER_BOUNDS = (21, 134), (46, 141)

# Maximum number of objects (from MAX_NB_OBJECTS)
MAX_DIVERS = 4
MAX_SHARKS = 12
MAX_SUBS = 12
MAX_ENEMY_MISSILES = 4
MAX_PLAYER_TORPS = 1
MAX_SURFACE_SUBS = 1
MAX_COLLECTED_DIVERS = 6

# define object orientations
FACE_LEFT = -1
FACE_RIGHT = 1

SPAWN_POSITIONS_Y = jnp.array([71, 95, 119, 139])  # submarines at y=69?
SUBMARINE_Y_OFFSET = 2
ENEMY_MISSILE_Y = jnp.array([73, 97, 121, 141])  # missile x = submarine.x + 4
DIVER_SPAWN_POSITIONS = jnp.array([69, 93, 117, 141])

MISSILE_SPAWN_POSITIONS = jnp.array([39, 126])  # Right, Left

# First wave directions from original code
FIRST_WAVE_DIRS = jnp.array([False, False, False, True])

class SpawnState(NamedTuple):
    difficulty: chex.Array  # Current difficulty level (0-7)
    lane_dependent_pattern: chex.Array  # Track waves independently per lane [4 lanes]
    to_be_spawned: (
        chex.Array
    )  # tracks which enemies are still in the spawning cycle [4 lanes * 3 slots] -> necessary due to the spaced out spawning of multiple enemies
    survived: (
        chex.Array
    )  # track if last enemy survived [4 lanes * 3 slots] -> 1 if survived whilst going right, 0 if not, -1 if survived whilst going left
    prev_sub: chex.Array  # Track previous entity type for each lane [4 lanes]
    spawn_timers: chex.Array  # Individual spawn timers per lane [4 lanes]
    diver_array: (
        chex.Array
    )  # Track which divers are still in the spawning cycle [4 lanes]
    lane_directions: (
        chex.Array
    )  # Track lane directions for each wave [4 lanes] -> 0 = right, 1 = left


def initialize_spawn_state() -> SpawnState:
    """Initialize spawn state with first wave matching original game."""
    return SpawnState(
        difficulty=jnp.array(0),
        lane_dependent_pattern=jnp.zeros(
            4, dtype=jnp.int32
        ),  # Each lane starts at wave 0
        to_be_spawned=jnp.zeros(
            12, dtype=jnp.int32
        ),  # Track which enemies are still in the spawning cycle
        survived=jnp.zeros(12, dtype=jnp.int32),  # Track which enemies survived
        prev_sub=jnp.ones(
            4, dtype=jnp.int32
        ),  # Track previous entity type (0 if shark, 1 if sub) -> starts at 1 since the first wave is sharks
        spawn_timers=jnp.array(
            [277, 277, 277, 277 + 60], dtype=jnp.int32
        ),  # All lanes start with same timer
        diver_array=jnp.array([1, 1, 0, 0], dtype=jnp.int32),
        lane_directions=FIRST_WAVE_DIRS.astype(jnp.int32),  # First wave directions
    )


def soft_reset_spawn_state(spawn_state: SpawnState) -> SpawnState:
    """Reset spawn_times"""
    return spawn_state._replace(
        spawn_timers=jnp.array([277, 277, 277, 277], dtype=jnp.int32)
    )

# Game state container
class SeaquestState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array  # 0 for right, 1 for left
    oxygen: chex.Array
    divers_collected: chex.Array
    score: chex.Array
    lives: chex.Array
    spawn_state: SpawnState
    diver_positions: chex.Array  # (4, 3) array for divers
    shark_positions: (
        chex.Array
    )  # (12, 3) array for sharks - separated into 4 lanes, 3 slots per lane [left to right]
    sub_positions: (
        chex.Array
    )  # (12, 3) array for enemy subs - separated into 4 lanes, 3 slots per lane [left to right]
    enemy_missile_positions: (
        chex.Array
    )  # (4, 3) array for enemy missiles (only the front boats can shoot)
    surface_sub_position: chex.Array  # (1, 3) array for surface submarine
    player_missile_position: (
        chex.Array
    )  # (1, 3) array for player missile (x, y, direction)
    step_counter: chex.Array
    just_surfaced: chex.Array  # Flag for tracking actual surfacing moment
    successful_rescues: (
        chex.Array
    )  # Number of times the player has surfaced with all six divers
    death_counter: chex.Array  # Counter for tracking death animation
    rng_key: chex.PRNGKey


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray


class SeaquestObservation(NamedTuple):
    player: EntityPosition
    sharks: jnp.ndarray  # Shape (12, 5) - 12 sharks, each with x,y,w,h,active
    submarines: jnp.ndarray  # Shape (12, 5)
    divers: jnp.ndarray  # Shape (4, 5)
    enemy_missiles: jnp.ndarray  # Shape (4, 5)
    surface_submarine: EntityPosition
    player_missile: EntityPosition
    collected_divers: jnp.ndarray  # Number of divers collected (0-6)
    player_score: jnp.ndarray
    lives: jnp.ndarray
    oxygen_level: jnp.ndarray  # Oxygen level (0-255)

class SeaquestInfo(NamedTuple):
    difficulty: jnp.ndarray  # Current difficulty level
    successful_rescues: jnp.ndarray  # Number of successful rescues
    step_counter: jnp.ndarray  # Current step count
    all_rewards: jnp.ndarray  # All rewards for the current step


class CarryState(NamedTuple):
    missile_pos: chex.Array
    shark_pos: chex.Array
    sub_pos: chex.Array
    score: chex.Array


# RENDER CONSTANTS
def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Load sprites - no padding needed for background since it's already full size
    bg1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/bg/1.npy"))
    pl_sub1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/player_sub/1.npy"))
    pl_sub2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/player_sub/2.npy"))
    pl_sub3 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/player_sub/3.npy"))
    diver1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/diver/1.npy"))
    diver2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/diver/2.npy"))
    shark1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/shark/1.npy"))
    shark2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/shark/2.npy"))
    enemy_sub1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/enemy_sub/1.npy"))
    enemy_sub2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/enemy_sub/2.npy"))
    enemy_sub3 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/enemy_sub/3.npy"))
    pl_torp = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/player_torp/1.npy"))
    en_torp = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/enemy_torp/1.npy"))

    # Pad player submarine sprites to match each other
    pl_sub_sprites = aj.pad_to_match([pl_sub1, pl_sub2, pl_sub3])

    # Pad diver sprites to match each other
    diver_sprites = aj.pad_to_match([diver1, diver2])

    # Pad shark sprites to match each other
    shark_sprites = aj.pad_to_match([shark1, shark2])

    # Pad enemy submarine sprites to match each other
    enemy_sub_sprites = aj.pad_to_match([enemy_sub1, enemy_sub2, enemy_sub3])

    # Pad player torpedo sprites to match each other
    pl_torp_sprites = [pl_torp]

    # Pad enemy torpedo sprites to match each other
    en_torp_sprites = [en_torp]

    # Background sprite (no padding needed)
    SPRITE_BG = jnp.expand_dims(bg1, axis=0)

    # Player submarine sprites
    SPRITE_PL_SUB = jnp.concatenate(
        [
            jnp.repeat(pl_sub_sprites[0][None], 4, axis=0),
            jnp.repeat(pl_sub_sprites[1][None], 4, axis=0),
            jnp.repeat(pl_sub_sprites[2][None], 4, axis=0),
        ]
    )

    # Diver sprites
    SPRITE_DIVER = jnp.concatenate(
        [
            jnp.repeat(diver_sprites[0][None], 16, axis=0),
            jnp.repeat(diver_sprites[1][None], 4, axis=0),
        ]
    )

    # Shark sprites
    SPRITE_SHARK = jnp.concatenate(
        [
            jnp.repeat(shark_sprites[0][None], 16, axis=0),
            jnp.repeat(shark_sprites[1][None], 8, axis=0),
        ]
    )

    # Enemy submarine sprites
    SPRITE_ENEMY_SUB = jnp.concatenate(
        [
            jnp.repeat(enemy_sub_sprites[0][None], 4, axis=0),
            jnp.repeat(enemy_sub_sprites[1][None], 4, axis=0),
            jnp.repeat(enemy_sub_sprites[2][None], 4, axis=0),
        ]
    )

    DIGITS = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "./sprites/seaquest/digits/{}.npy"))
    LIFE_INDICATOR = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/life_indicator/1.npy"))
    DIVER_INDICATOR = aj.loadFrame(os.path.join(MODULE_DIR, "./sprites/seaquest/diver_indicator/1.npy"))

    # Player torpedo sprites
    SPRITE_PL_TORP = jnp.repeat(pl_torp_sprites[0][None], 1, axis=0)

    # Enemy torpedo sprites
    SPRITE_EN_TORP = jnp.repeat(en_torp_sprites[0][None], 1, axis=0)

    return (
        SPRITE_BG,
        SPRITE_PL_SUB,
        SPRITE_DIVER,
        SPRITE_SHARK,
        SPRITE_ENEMY_SUB,
        SPRITE_PL_TORP,
        SPRITE_EN_TORP,
        DIGITS,
        LIFE_INDICATOR,
        DIVER_INDICATOR,
    )


# Load sprites once at module level
(
    SPRITE_BG,
    SPRITE_PL_SUB,
    SPRITE_DIVER,
    SPRITE_SHARK,
    SPRITE_ENEMY_SUB,
    SPRITE_PL_TORP,
    SPRITE_EN_TORP,
    DIGITS,
    LIFE_INDICATOR,
    DIVER_INDICATOR,
) = load_sprites()

@jax.jit
def check_collision_single(pos1, size1, pos2, size2):
    """Check collision between two single entities"""
    # Calculate edges for rectangle 1
    rect1_left = pos1[0]
    rect1_right = pos1[0] + size1[0]
    rect1_top = pos1[1]
    rect1_bottom = pos1[1] + size1[1]

    # Calculate edges for rectangle 2
    rect2_left = pos2[0]
    rect2_right = pos2[0] + size2[0]
    rect2_top = pos2[1]
    rect2_bottom = pos2[1] + size2[1]

    # Check overlap
    horizontal_overlap = jnp.logical_and(
        rect1_left < rect2_right,
        rect1_right > rect2_left
    )

    vertical_overlap = jnp.logical_and(
        rect1_top < rect2_bottom,
        rect1_bottom > rect2_top
    )

    return jnp.logical_and(horizontal_overlap, vertical_overlap)

@jax.jit
def check_collision_batch(pos1, size1, pos2_array, size2):
    """Check collision between one entity and an array of entities"""
    # Calculate edges for rectangle 1
    rect1_left = pos1[0]
    rect1_right = pos1[0] + size1[0]
    rect1_top = pos1[1]
    rect1_bottom = pos1[1] + size1[1]

    # Calculate edges for all rectangles in pos2_array
    rect2_left = pos2_array[:, 0]
    rect2_right = pos2_array[:, 0] + size2[0]
    rect2_top = pos2_array[:, 1]
    rect2_bottom = pos2_array[:, 1] + size2[1]

    # Check overlap for all entities
    horizontal_overlaps = jnp.logical_and(
        rect1_left < rect2_right,
        rect1_right > rect2_left
    )

    vertical_overlaps = jnp.logical_and(
        rect1_top < rect2_bottom,
        rect1_bottom > rect2_top
    )

    # Combine checks for each entity
    collisions = jnp.logical_and(horizontal_overlaps, vertical_overlaps)

    # Return true if any collision detected
    return jnp.any(collisions)

@jax.jit
def check_missile_collisions(
    missile_pos: chex.Array,
    shark_positions: chex.Array,
    sub_positions: chex.Array,
    score: chex.Array,
    successful_rescues: chex.Array,
    spawn_state: SpawnState,
    rng_key: chex.PRNGKey,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, SpawnState, chex.PRNGKey]:
    """Check for collisions between player missile and enemies"""
    # Create missile position array for collision check
    missile_rect_pos = jnp.array([missile_pos[0], missile_pos[1]])
    missile_active = missile_pos[2] != 0

    # Split RNG for collision detection and direction randomization
    rng_key, direction_rng = jax.random.split(rng_key)

    def check_enemy_collisions(enemy_idx, carry):
        # Unpack carry state
        missile_pos, shark_positions, sub_positions, score, spawn_state = carry

        # Check shark collisions - only if missile is active
        shark_collision = jnp.logical_and(
            missile_active,
            check_collision_single(  # Use single version
                missile_rect_pos, MISSILE_SIZE,
                shark_positions[enemy_idx], SHARK_SIZE
            )
        )

        # Check submarine collisions - only if missile is active
        sub_collision = jnp.logical_and(
            missile_active,
            check_collision_single(  # Use single version
                missile_rect_pos, MISSILE_SIZE,
                sub_positions[enemy_idx], ENEMY_SUB_SIZE
            )
        )

        # Update positions and score - use where instead of if statements
        new_shark_pos = jnp.where(
            shark_collision,
            jnp.zeros_like(shark_positions[enemy_idx]),
            shark_positions[enemy_idx],
        )

        new_sub_pos = jnp.where(
            sub_collision,
            jnp.zeros_like(sub_positions[enemy_idx]),
            sub_positions[enemy_idx],
        )

        # Update score
        score_increase = jnp.where(
            shark_collision,
            calculate_kill_points(successful_rescues),
            jnp.where(sub_collision, calculate_kill_points(successful_rescues), 0),
        )

        # Remove missile if it hit anything
        new_missile_pos = jnp.where(
            jnp.logical_or(shark_collision, sub_collision),
            jnp.array([0, 0, 0]),
            missile_pos,
        )

        # Update the kill tracking in spawn state
        any_collision = jnp.logical_or(shark_collision, sub_collision)
        new_survived = spawn_state.survived.at[enemy_idx].set(
            jnp.where(
                any_collision,
                0,  # Set to False when destroyed by missile
                spawn_state.survived[enemy_idx],
            )
        )

        # Update spawn timers when enemy destroyed
        lane_idx = (enemy_idx // 3).astype(int)
        new_spawn_timers = spawn_state.spawn_timers.at[lane_idx].set(
            jnp.where(
                any_collision, 200, spawn_state.spawn_timers[lane_idx].astype(int)
            )
        )

        new_lane_directions = spawn_state.lane_directions
        # if survived is true, also randomize the lane direction for the next spawn cycle -> this might lead to some unnecessary randomization, but the impact on performance should be negligible
        new_lane_directions = jnp.where(
            any_collision,
            new_lane_directions.at[enemy_idx // 3].set(
                jax.random.bernoulli(direction_rng, 0.5)
            ),
            new_lane_directions,
        )

        # Create updated spawn state (including lane_directions field)
        new_spawn_state = SpawnState(
            difficulty=spawn_state.difficulty,
            lane_dependent_pattern=spawn_state.lane_dependent_pattern,
            to_be_spawned=spawn_state.to_be_spawned,
            spawn_timers=new_spawn_timers,
            prev_sub=spawn_state.prev_sub,
            survived=new_survived,
            diver_array=spawn_state.diver_array,
            lane_directions=new_lane_directions,
        )

        return (
            new_missile_pos,
            shark_positions.at[enemy_idx].set(new_shark_pos),
            sub_positions.at[enemy_idx].set(new_sub_pos),
            score + score_increase,
            new_spawn_state,
        )

    # Run collision detection for each possible enemy
    missile_pos, shark_positions, sub_positions, score, spawn_state = jax.lax.fori_loop(
        0,
        shark_positions.shape[0],
        check_enemy_collisions,
        (missile_pos, shark_positions, sub_positions, score, spawn_state),
    )

    return (
        missile_pos,
        shark_positions,
        sub_positions,
        score,
        spawn_state,
        direction_rng,
    )

@jax.jit
def check_player_collision(
    player_x,
    player_y,
    submarine_list,
    shark_list,
    surface_sub_pos,
    enemy_projectile_list,
    score,
    successful_rescues,
) -> Tuple[chex.Array, chex.Array]:
    # check if the player has collided with any of the three given lists
    # the player is a 16x11 rectangle
    # the submarine is a 8x11 rectangle
    # the shark is a 8x7 rectangle
    # the missile is a 8x1 rectangle
    # the surface submarine is 8x11 as well

    # check if the player has collided with any of the submarines
    submarine_collisions = jnp.any(
        check_collision_batch(
            jnp.array([player_x, player_y]), PLAYER_SIZE, submarine_list, ENEMY_SUB_SIZE
        )
    )

    # check if the player has collided with any of the sharks
    shark_collisions = jnp.any(
        check_collision_batch(
            jnp.array([player_x, player_y]), PLAYER_SIZE, shark_list, SHARK_SIZE
        )
    )

    # check if the player collided with the surface submarine
    surface_collision = check_collision_single(
        jnp.array([player_x, player_y]),
        PLAYER_SIZE,
        surface_sub_pos,
        ENEMY_SUB_SIZE
    )

    # check if the player has collided with any of the enemy projectiles
    missile_collisions = jnp.any(
        check_collision_batch(
            jnp.array([player_x, player_y]),
            PLAYER_SIZE,
            enemy_projectile_list,
            MISSILE_SIZE
        )
    )

    # Calculate points for collisions.
    # When colliding with a shark or submarine the player gains points similar to killing the object
    collision_points = jnp.where(
        shark_collisions,
        calculate_kill_points(successful_rescues),
        jnp.where(
            submarine_collisions,
            calculate_kill_points(successful_rescues),
            jnp.where(surface_collision, calculate_kill_points(successful_rescues), 0),
        ),
    )

    return (
        jnp.any(
            jnp.array(
                [
                    submarine_collisions,
                    shark_collisions,
                    missile_collisions,
                    surface_collision,
                ]
            )
        ),
        collision_points,
    )

@jax.jit
def get_spawn_position(moving_left: chex.Array, slot: chex.Array) -> chex.Array:
    """Get spawn position based on movement direction and slot number"""
    base_y = jnp.array(SPAWN_POSITIONS_Y[slot])
    x_pos = jnp.where(
        moving_left,
        jnp.array(165, dtype=jnp.int32),  # Start right if moving left
        jnp.array(0, dtype=jnp.int32),
    )  # Start left if moving right
    direction = jnp.where(moving_left, -1, 1)  # -1 for left, 1 for right
    return jnp.array([x_pos, base_y, direction], dtype=jnp.int32)

@jax.jit
def is_slot_empty(pos: chex.Array) -> chex.Array:
    """Check if a position slot is empty (0,0,ß)"""
    return pos[2] == 0

@jax.jit
def get_front_entity(i, lane_positions):
    # check on the first submarine in the lane which direction they are going
    direction = lane_positions[0][2]

    direction = jnp.where(
        lane_positions[0][2] == 0,
        jnp.where(
            lane_positions[1][2] == 0, lane_positions[2][2], lane_positions[1][2]
        ),
        lane_positions[0][2],
    )

    # if direction is 1, go from right to left until an active entity is found
    # if direction is -1, go from left to right until an active entity is found
    front_entity = jnp.where(
        direction == -1,
        jnp.where(
            lane_positions[0][2] != 0,
            lane_positions[0],
            jnp.where(
                lane_positions[1][2] != 0,
                lane_positions[1],
                jnp.where(lane_positions[2][2] != 0, lane_positions[2], jnp.zeros(3)),
            ),
        ),
        jnp.where(
            lane_positions[2][2] != 0,
            lane_positions[2],
            jnp.where(
                lane_positions[1][2] != 0,
                lane_positions[1],
                jnp.where(lane_positions[0][2] != 0, lane_positions[0], jnp.zeros(3)),
            ),
        ),
    )

    return front_entity

@jax.jit
def get_pattern_for_difficulty(
    current_pattern: chex.Array, moving_left: chex.Array
) -> chex.Array:
    """Returns spawn pattern based on the lane's current wave/pattern number

    Pattern meanings:
    0: Single enemy (initial pattern)
    1: Two adjacent enemies
    2: Two enemies with gap
    3: Three enemies in a row
    """
    # Basic pattern arrays for different formations
    PATTERNS = jnp.array(
        [
            [0, 0, 1],  # wave 0: Single enemy
            [0, 1, 1],  # wave 1: Two adjacent
            [1, 0, 1],  # wave 2: Two with gap
            [1, 1, 1],  # wave 3: Three in row
        ]
    )

    # Reverse pattern if moving left
    base_pattern = PATTERNS[current_pattern]

    return base_pattern

@jax.jit
def update_enemy_spawns(
    spawn_state: SpawnState,
    shark_positions: chex.Array,
    sub_positions: chex.Array,
    diver_positions: chex.Array,
    step_counter: chex.Array,
    rng: chex.PRNGKey = None,
) -> Tuple[SpawnState, chex.Array, chex.Array, chex.PRNGKey]:
    """Update enemy spawns using pattern-based system matching original game.
    Args:
        spawn_state: Current spawn state
        shark_positions: Current shark positions
        sub_positions: Current submarine positions
        diver_positions: Current diver positions
        step_counter: Current step counter
        rng: Optional random key for direction randomization

    Returns:
        Tuple of updated spawn state, shark positions, sub positions, and updated RNG key
    """

    # count down the spawn timers
    new_spawn_timers = jnp.where(
        spawn_state.spawn_timers > 0,
        spawn_state.spawn_timers - 1,
        spawn_state.spawn_timers,
    )

    new_state = spawn_state._replace(spawn_timers=new_spawn_timers)

    # Define a function for jax.lax.scan to process each lane
    def scan_lanes(carry, lane_idx):
        curr_state, curr_shark_positions, curr_sub_positions, curr_diver_positions, curr_rng = carry

        # Check if this lane needs an update
        needs_update = lane_needs_update(lane_idx, curr_state, curr_shark_positions, curr_sub_positions)

        # Process the lane if needed
        new_carry = jax.lax.cond(
            needs_update,
            lambda x: process_lane(lane_idx, x),
            lambda x: x,
            (curr_state, curr_shark_positions, curr_sub_positions, curr_diver_positions, curr_rng),
        )

        return new_carry, None  # None for outputs as we only care about final state

    def initialize_new_spawn_cycle(i, carry):
        spawn_state, shark_positions, sub_positions, diver_positions, rng = carry

        # Split RNG key for this lane
        rng, lane_rng = jax.random.split(rng)

        # Get survived status for this lane (3 slots)
        lane_survived = jax.lax.dynamic_slice(spawn_state.survived, (i * 3,), (3,))

        # Update the difficulty patterns for this lane
        left_over = jnp.any(lane_survived)
        clipped_difficulty = spawn_state.difficulty % 8
        # Update spawn state
        lane_specific_pattern = jnp.where(
            jnp.logical_not(left_over),  # Only update if all destroyed
            jnp.where(
                clipped_difficulty < 2,
                0,
                jnp.where(
                    clipped_difficulty < 4,
                    1,
                    jnp.where(
                        clipped_difficulty < 6,
                        2,
                        jnp.where(clipped_difficulty < 8, 3, 0),
                    ),
                ),
            ),
            spawn_state.lane_dependent_pattern[i],
        )

        # Check if there's an active diver in this lane
        active_diver = diver_positions[i][2] != 0
        diver_direction = diver_positions[i][2]

        # If there's an active diver, use its direction, otherwise randomize
        moving_left = jnp.where(
            active_diver,
            diver_direction == -1,  # Use diver's direction if active
            spawn_state.lane_directions[i] == 1  # Otherwise use current lane direction
        )

        # get the spawn pattern for this lane
        # Check if this slot had something survive last time (if yes, we have to overwrite the current_pattern)
        current_pattern = jnp.where(
            left_over,
            lane_survived,
            get_pattern_for_difficulty(lane_specific_pattern, moving_left),
        )

        # make sure that in the current pattern all entries are positive (i.e. abs() on all values)
        current_pattern = jnp.abs(current_pattern)

        # in case we are going left, flip the pattern
        current_pattern = jnp.where(
            moving_left, -jnp.flip(current_pattern), current_pattern
        )

        # check if this should be a submarine or a shark
        is_sub = jnp.logical_and(left_over, jnp.logical_not(spawn_state.prev_sub[i]))

        # set the positions for the first enemy in the wave (dependent on the direction this is either the first or the last slot)
        first_slot = jnp.where(moving_left, 0, 2)

        base_pos = get_spawn_position(moving_left, jnp.array(i))
        # spawn the first enemy in the wave
        new_shark_positions = jnp.where(
            is_sub,
            shark_positions,
            shark_positions.at[(i * 3 + first_slot)].set(base_pos),
        )

        new_sub_positions = jnp.where(
            is_sub, sub_positions.at[(i * 3 + first_slot)].set(base_pos), sub_positions
        )

        # wipe the survived status for this lane (since we are starting a new wave)
        indices = jnp.array([i * 3, i * 3 + 1, i * 3 + 2])
        new_survived_full = spawn_state.survived.at[indices].set(
            jnp.zeros(3, dtype=jnp.int32)
        )

        # Set moving_left to the opposite of moving_left when determining which slot to clear in to_be_spawned
        new_to_be_spawned = current_pattern.at[jnp.where(moving_left, 0, 2)].set(0)

        # Update the full to_be_spawned array for this lane
        new_full_to_be_spawned = spawn_state.to_be_spawned.at[indices].set(
            new_to_be_spawned
        )

        new_spawn_state = SpawnState(
            difficulty=spawn_state.difficulty,
            lane_dependent_pattern=spawn_state.lane_dependent_pattern.at[i].set(
                lane_specific_pattern
            ),
            to_be_spawned=new_full_to_be_spawned,
            survived=new_survived_full,
            prev_sub=spawn_state.prev_sub.at[i].set(is_sub),
            spawn_timers=spawn_state.spawn_timers.at[i].set(200),
            diver_array=spawn_state.diver_array,
            lane_directions=spawn_state.lane_directions,
        )

        return new_spawn_state, new_shark_positions, new_sub_positions, diver_positions, rng

    # Modified continue_spawn_cycle to handle RNG
    def continue_spawn_cycle(i: int, carry):
        spawn_state, shark_positions, sub_positions, diver_positions, rng = carry

        # Rest of function remains the same, just pass along the RNG
        # get the relevant missing entities for this lane from the to_be_spawned array
        relevant_to_be_spawned = jax.lax.dynamic_slice(
            spawn_state.to_be_spawned, (i * 3,), (3,)
        )

        # check in which direction we are moving by finding the first non-zero value in the missing_entities array
        moving_left = jnp.where(
            relevant_to_be_spawned[0] == 0,
            jnp.where(
                relevant_to_be_spawned[1] == 0,
                jnp.where(relevant_to_be_spawned[2] == -1, True, False),
                jnp.where(relevant_to_be_spawned[1] == -1, True, False),
            ),
            jnp.where(relevant_to_be_spawned[0] == -1, True, False),
        )

        # Find the index of the first non-zero value based on direction
        def scan_right_to_left(j, val):
            return jnp.where(relevant_to_be_spawned[2 - j] != 0, 2 - j, val)

        def scan_left_to_right(j, val):
            return jnp.where(relevant_to_be_spawned[j] != 0, j, val)

        # Use fori_loop to scan array in appropriate direction
        spawn_idx = jax.lax.cond(
            moving_left,
            lambda _: jax.lax.fori_loop(0, 3, scan_left_to_right, -1),
            lambda _: jax.lax.fori_loop(0, 3, scan_right_to_left, -1),
            operand=None,
        )

        spawn_idx = spawn_idx.astype(jnp.int32)

        # Get reference x position from neighboring entity
        # For moving right, look at entity to the right (spawn_idx + 1)
        # For moving left, look at entity to the left (spawn_idx - 1)
        reference_idx = jnp.where(moving_left, spawn_idx - 1, spawn_idx + 1)
        reference_idx = reference_idx.astype(jnp.int32)
        base_idx = i * 3  # Base index for this lane's entities

        # Get position from either shark or sub position arrays
        # We'll need to check both since we don't know which type exists
        reference_shark_pos = shark_positions[base_idx + reference_idx]
        reference_sub_pos = sub_positions[base_idx + reference_idx]

        # Use whichever position is non-zero (active)
        reference_x = jnp.where(
            reference_shark_pos[0] != 0, reference_shark_pos[0], reference_sub_pos[0]
        )

        edge_case = reference_x == 0
        # Edge Case: third option exists for the pattern 1 0 1, then check the next entity
        edge_case_reference_idx = jnp.where(moving_left, spawn_idx - 2, spawn_idx + 2)

        edge_case_reference_idx = edge_case_reference_idx.astype(jnp.int32)

        reference_x = jnp.where(
            edge_case,
            jnp.where(
                shark_positions[base_idx + edge_case_reference_idx][0] != 0,
                shark_positions[base_idx + edge_case_reference_idx][0],
                sub_positions[base_idx + edge_case_reference_idx][0],
            ),
            reference_x,
        )

        # Get base spawn position for this lane
        base_spawn_pos = get_spawn_position(moving_left, jnp.array(i))

        # check if the base spawn position x is 16 / 32 pixels away from the reference x position (depending on the edge case pattern)
        # if yes, spawn the entity, if no, do nothing
        offset = jnp.where(edge_case, 32, 16)
        should_spawn = jnp.abs(base_spawn_pos[0] - reference_x) >= offset

        # in case reference_x is still 0 (happens in case the player destroyed the first entity in the wave), we just instantly spawn the entity
        should_spawn = jnp.where(reference_x == 0, True, should_spawn)

        spawn_pos = jnp.where(should_spawn, base_spawn_pos, jnp.zeros(3))

        # Update positions based on enemy type
        new_shark_positions = shark_positions.at[base_idx + spawn_idx].set(
            jnp.where(
                jnp.logical_not(spawn_state.prev_sub[i]),
                spawn_pos,
                shark_positions[base_idx + spawn_idx],
            )
        )
        new_sub_positions = sub_positions.at[base_idx + spawn_idx].set(
            jnp.where(
                spawn_state.prev_sub[i], spawn_pos, sub_positions[base_idx + spawn_idx]
            )
        )

        # Update the to_be_spawned array
        new_to_be_spawned = spawn_state.to_be_spawned.at[base_idx + spawn_idx].set(
            jnp.where(
                should_spawn,
                jnp.array(0),  # Single value
                spawn_state.to_be_spawned[base_idx + spawn_idx],
            )
        )

        # Then create the new spawn state with the updated array
        new_spawn_state = SpawnState(
            difficulty=spawn_state.difficulty,
            lane_dependent_pattern=spawn_state.lane_dependent_pattern,
            to_be_spawned=new_to_be_spawned,
            survived=spawn_state.survived,
            prev_sub=spawn_state.prev_sub,
            spawn_timers=spawn_state.spawn_timers,
            diver_array=spawn_state.diver_array,
            lane_directions=spawn_state.lane_directions,
        )

        return new_spawn_state, new_shark_positions, new_sub_positions, diver_positions, rng

    # Modified process_lane to handle RNG
    def process_lane(i, carry):
        loc_spawn_state, shark_positions, sub_positions, diver_positions, rng = carry
        base_idx = i * 3  # Base index for this lane's slots

        # determine if we need to initialize a new pattern or keep spawning for the current one
        # do this by checking in the relevant part of the to_be_spawned array if there are still 1s
        relevant_to_be_spawned = jax.lax.dynamic_slice(
            spawn_state.to_be_spawned, (base_idx,), (3,)
        )

        # if there are still 1s in the relevant part of the to_be_spawned array, keep spawning
        keep_spawning = jnp.any(relevant_to_be_spawned)

        # check the lane spawn timer
        lane_timer = spawn_state.spawn_timers[i]

        lane_empty = jnp.all(
            jnp.array(
                [
                    jnp.logical_and(
                        is_slot_empty(shark_positions[base_idx + j]),
                        is_slot_empty(sub_positions[base_idx + j]),
                    )
                    for j in range(3)
                ]
            )
        )

        # if the lane timer is unequal to 0, continue_spawn_cycle may still be called but initialize_new_spawn_cycle should not be called
        allow_new_initialization = jnp.logical_and(lane_timer == 0, lane_empty)

        def handle_no_spawning(x):
            spawn_state, shark_positions, sub_positions, diver_positions, rng = x
            return jax.lax.cond(
                allow_new_initialization,
                lambda y: initialize_new_spawn_cycle(i, y),
                lambda y: (y[0], y[1], y[2], y[3], y[4]),  # Return unchanged state
                (spawn_state, shark_positions, sub_positions, diver_positions, rng),
            )

        new_spawn_state, new_shark_positions, new_sub_positions, new_diver_positions, new_rng = jax.lax.cond(
            keep_spawning,
            lambda x: continue_spawn_cycle(i, x),
            handle_no_spawning,
            (loc_spawn_state, shark_positions, sub_positions, diver_positions, rng),
        )

        return new_spawn_state, new_shark_positions, new_sub_positions, new_diver_positions, new_rng

    # Modify lane_needs_update to work with the rest of the function
    def lane_needs_update(i, spawn_state, shark_positions, sub_positions):
        base_idx = i * 3  # Base index for this lane's slots

        # get how many entities in this lane are inactive
        lane_empty = jnp.all(
            jnp.array(
                [
                    jnp.logical_and(
                        is_slot_empty(shark_positions[base_idx + j]),
                        is_slot_empty(sub_positions[base_idx + j]),
                    )
                    for j in range(3)
                ]
            )
        )

        # check if the to_be_spawned array has any 1s in the relevant part
        relevant_to_be_spawned = jax.lax.dynamic_slice(
            spawn_state.to_be_spawned, (base_idx,), (3,)
        )

        return jnp.logical_or(lane_empty, jnp.any(relevant_to_be_spawned))

    # Replace the manual loop with lax.scan
    lane_indices = jnp.arange(4)
    (final_state, final_shark_positions, final_sub_positions, final_diver_positions, final_rng), _ = jax.lax.scan(
        scan_lanes,
        (new_state, shark_positions, sub_positions, diver_positions, rng if rng is not None else jax.random.PRNGKey(42)),
        lane_indices
    )

    return final_state, final_shark_positions, final_sub_positions, final_rng

@jax.jit
def step_enemy_movement(
    spawn_state: SpawnState,
    shark_positions: chex.Array,
    sub_positions: chex.Array,
    step_counter: chex.Array,
    rng: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array, SpawnState, chex.PRNGKey]:
    """Update enemy positions based on their patterns"""
    # Split RNG key for direction randomization
    rng, direction_rng = jax.random.split(rng)

    def get_shark_offset(
        step_counter,
    ):  # shark offset should be constant over the difficulty levels..
        phase = step_counter // 4
        cycle_position = phase % 32

        raw_offset = jnp.where(
            cycle_position < 16,
            cycle_position // 2,  # 0->7
            7 - (cycle_position - 16) // 2,  # 7->0
        )

        return raw_offset - 4

    def calculate_movement_speed(step_counter, difficulty):
        """Calculate movement speed with JIT-compatible operations.
        Uses array indexing and jnp.select for cleaner, switch-case-like behavior.

        Args:
            step_counter: Current step counter
            difficulty: Current difficulty level (0-255)

        Returns:
            Movement speed for the current frame
        """
        cycle_pos = step_counter % 12

        # Handling difficulties 0-9 using array lookup

        # Ensure difficulty is non-negative (safety check)
        safe_difficulty = jnp.maximum(0, difficulty)

        # Create a boolean array for each difficulty bracket (0-9)
        diff_brackets = jnp.array(
            [
                safe_difficulty == 0,  # Difficulty 0
                jnp.logical_and(
                    safe_difficulty >= 1, safe_difficulty <= 2
                ),  # Difficulty 1-2
                jnp.logical_and(
                    safe_difficulty >= 3, safe_difficulty <= 4
                ),  # Difficulty 3-4
                jnp.logical_and(
                    safe_difficulty >= 5, safe_difficulty <= 6
                ),  # Difficulty 5-6
                jnp.logical_and(
                    safe_difficulty >= 7, safe_difficulty <= 8
                ),  # Difficulty 7-8
                safe_difficulty == 9,  # Difficulty 9
            ]
        )

        # Create an array of movement patterns
        should_move_patterns = jnp.array(
            [
                (cycle_pos % 3) == 0,  # Difficulty 0: 33% movement (1 in 3 frames)
                (cycle_pos % 2) == 0,  # Difficulty 1-2: 50% movement (1 in 2 frames)
                (cycle_pos % 3) != 2,  # Difficulty 3-4: 67% movement (2 in 3 frames)
                (cycle_pos % 4) != 3,  # Difficulty 5-6: 75% movement (3 in 4 frames)
                (cycle_pos % 6) != 5,  # Difficulty 7-8: 83% movement (5 in 6 frames)
                cycle_pos != 11,  # Difficulty 9: 92% movement (11 in 12 frames)
            ]
        )

        # Use jnp.select to choose the correct movement pattern (like a switch-case)
        # Default to False to ensure predictable behavior for unexpected difficulty values
        should_move = jnp.select(diff_brackets, should_move_patterns, default=False)

        # For difficulties 0-9, return 1 if should move, 0 otherwise
        speed_for_diff_0_9 = jnp.where(should_move, 1, 0)

        # For difficulty 10+
        # Handle wrapping at difficulty 255
        adjusted_difficulty = difficulty % 256

        # Handle difficulty 10+ with tier-based speeds
        # For difficulties < 10, these calculations aren't used
        diff_above_threshold = jnp.maximum(0, adjusted_difficulty - 10)

        # Base speed calculation (tier-based)
        # Difficulty 10-25: Base speed 1
        # Difficulty 26-41: Base speed 2, etc.
        base_speed = 1 + (diff_above_threshold // 16)

        # Calculate position within the current tier (0-15)
        position_in_tier = diff_above_threshold % 16

        # Create position bracket array (much cleaner than nested where statements)
        pos_brackets = jnp.array(
            [
                position_in_tier == 0,  # Position 0
                jnp.logical_and(
                    position_in_tier >= 1, position_in_tier <= 3
                ),  # Position 1-3
                jnp.logical_and(
                    position_in_tier >= 4, position_in_tier <= 6
                ),  # Position 4-6
                jnp.logical_and(
                    position_in_tier >= 7, position_in_tier <= 9
                ),  # Position 7-9
                jnp.logical_and(
                    position_in_tier >= 10, position_in_tier <= 12
                ),  # Position 10-12
                jnp.logical_and(
                    position_in_tier >= 13, position_in_tier <= 14
                ),  # Position 13-14
                position_in_tier == 15,  # Position 15
            ]
        )

        # Create array of higher speed patterns
        higher_speed_patterns = jnp.array(
            [
                (step_counter % 16) == 0,  # Position 0: 1 in 16 frames (6.25%)
                (step_counter % 8) == 0,  # Position 1-3: 1 in 8 frames (12.5%)
                (step_counter % 4) == 0,  # Position 4-6: 1 in 4 frames (25%)
                (step_counter % 2) == 0,  # Position 7-9: 1 in 2 frames (50%)
                (step_counter % 4) != 0,  # Position 10-12: 3 in 4 frames (75%)
                (step_counter % 8) != 0,  # Position 13-14: 7 in 8 frames (87.5%)
                (step_counter % 16) != 0,  # Position 15: 15 in 16 frames (93.75%)
            ]
        )

        # Use jnp.select to choose the pattern (like a switch-case)
        use_higher_speed = jnp.select(
            pos_brackets, higher_speed_patterns, default=False
        )

        # Higher speed is base_speed + 1
        higher_speed = base_speed + 1

        # Speed for difficulty 10+: either base_speed or higher_speed
        speed_for_diff_10_plus = jnp.where(use_higher_speed, higher_speed, base_speed)

        # Return appropriate speed based on difficulty
        # Use safe_difficulty to ensure consistent behavior with negative inputs
        return jnp.where(
            safe_difficulty < 10, speed_for_diff_0_9, speed_for_diff_10_plus
        )

    def move_enemy(pos, is_shark, difficulty, slot_idx, step_counter):
        """Move enemy based on difficulty and pattern.

        Args:
            pos: Current position (x, y, direction)
            is_shark: Boolean indicating if this is a shark
            difficulty: Current difficulty level (0-255)
            slot_idx: Slot index (for lane determination)
            step_counter: Current step counter

        Returns:
            New position and whether the enemy is out of bounds
        """
        is_active = jnp.logical_not(is_slot_empty(pos))
        moving_left = pos[2] < 0

        # Calculate movement speed for this frame
        movement_speed = calculate_movement_speed(step_counter, difficulty)

        # Apply direction
        velocity_x = jnp.where(moving_left, -movement_speed, movement_speed)

        # Base Y position comes from spawn positions
        base_y = SPAWN_POSITIONS_Y[slot_idx // 3]  # Divide by 3 to get lane index

        # Calculate Y position
        y_position = jnp.where(
            is_shark,
            base_y + get_shark_offset(step_counter),
            # Submarines are 2 pixels higher than their base position
            base_y - SUBMARINE_Y_OFFSET,
        )

        # Apply movements
        new_pos = jnp.where(
            is_active, jnp.array([pos[0] + velocity_x, y_position, pos[2]]), pos
        )

        # Check bounds
        out_of_bounds = jnp.logical_or(new_pos[0] <= -8, new_pos[0] >= 168)
        return jnp.where(out_of_bounds, jnp.zeros_like(new_pos), new_pos), out_of_bounds

    new_shark_positions = jnp.zeros_like(shark_positions)
    new_sub_positions = jnp.zeros_like(sub_positions)
    new_survived = spawn_state.survived

    survived_dtype = spawn_state.survived.dtype
    new_lane_directions = spawn_state.lane_directions

    def process_shark_lane(carry, lane_idx):
        new_shark_positions, new_survived, new_lane_directions, new_diver_array, new_spawn_timers, direction_rng = carry

        # Static indices relative to lane start
        base_idx = lane_idx * 3
        rel_indices = jnp.array([0, 1, 2])  # These are fixed/static

        # Get direction of first shark in lane
        lane_sharks = jax.lax.dynamic_slice(shark_positions, (base_idx, 0), (3, 3))

        direction_of_shark = jnp.where(
            lane_sharks[0, 2] == 0,
            jnp.where(
                lane_sharks[1, 2] == 0,
                lane_sharks[2, 2],
                lane_sharks[1, 2]
            ),
            lane_sharks[0, 2]
        ).astype(survived_dtype)

        # Create slot indices by adding base_idx to each relative index
        slot_indices = base_idx + rel_indices

        # Process all sharks in this lane simultaneously
        def process_shark(shark_pos, slot_idx):
            new_pos, survived = move_enemy(
                shark_pos, True, spawn_state.difficulty, slot_idx, step_counter
            )
            survived_value = jnp.where(
                survived, direction_of_shark, new_survived[slot_idx]
            ).astype(survived_dtype)

            return new_pos, survived_value

        # Process all sharks in this lane simultaneously
        new_positions, survived_values = jax.vmap(process_shark)(
            lane_sharks, slot_indices
        )

        # Update positions using dynamic_update_slice instead of .at indexing with dynamic indices
        new_shark_positions = jax.lax.dynamic_update_slice(
            new_shark_positions, new_positions, (base_idx, 0)
        )

        # For survived values, we need to reshape to match dimensions
        # (assuming survived values is a 1D array of length 3)
        new_survived = jax.lax.dynamic_update_slice(
            new_survived, survived_values, (base_idx,)
        )

        # Get survived status for lane
        lane_survived = survived_values  # We already have this from above

        # Randomize lane direction if any shark survived
        direction_rng, next_rng = jax.random.split(direction_rng)
        new_lane_directions = jnp.where(
            jnp.any(lane_survived != 0),
            new_lane_directions.at[lane_idx].set(
                jnp.where(
                    jax.random.bernoulli(direction_rng, 0.5),
                    jnp.array(1, dtype=survived_dtype),
                    jnp.array(-1, dtype=survived_dtype)
                )
            ),
            new_lane_directions
        )

        # Direction flipping logic
        new_lane_survived = jnp.where(
            direction_of_shark == -1,
            jnp.flip(lane_survived),
            lane_survived
        )

        # Update lane survived status
        new_survived = jax.lax.dynamic_update_slice(
            new_survived, new_lane_survived, (base_idx,)
        )

        # Get survived status for lane
        lane_survived = survived_values  # We already have this from above

        # Check if any shark in the lane survived in THIS frame
        # Compare new survived values with old ones to detect changes
        old_survived = jax.lax.dynamic_slice(spawn_state.survived, (base_idx,), (3,))
        any_new_survived = jnp.any(jnp.logical_and(lane_survived != 0, old_survived == 0))

        # Update diver_array if needed
        new_diver_array = jnp.where(
            jnp.logical_and(any_new_survived, new_diver_array[lane_idx] == -1),
            new_diver_array.at[lane_idx].set(1),
            new_diver_array
        )

        # Reset spawn timer only when an enemy survives in this frame
        new_spawn_timers = new_spawn_timers.at[lane_idx].set(
            jnp.where(
                any_new_survived,
                200,  # Reset to 200 when an enemy survives in this frame
                new_spawn_timers[lane_idx]
            )
        )

        return (new_shark_positions, new_survived, new_lane_directions, new_diver_array, new_spawn_timers, next_rng), None

    # Similar changes for process_sub_lane
    def process_sub_lane(carry, lane_idx):
        new_sub_positions, new_survived, new_lane_directions, new_diver_array, new_spawn_timers, direction_rng = carry

        # Static indices relative to lane start
        base_idx = lane_idx * 3
        rel_indices = jnp.array([0, 1, 2])  # These are fixed/static

        # Get direction of first sub in lane
        lane_subs = jax.lax.dynamic_slice(sub_positions, (base_idx, 0), (3, 3))

        direction_of_sub = jnp.where(
            lane_subs[0, 2] == 0,
            jnp.where(
                lane_subs[1, 2] == 0,
                lane_subs[2, 2],
                lane_subs[1, 2]
            ),
            lane_subs[0, 2]
        ).astype(survived_dtype)

        # Create slot indices by adding base_idx to each relative index
        slot_indices = base_idx + rel_indices

        def process_sub(sub_pos, slot_idx):
            new_pos, survived = move_enemy(
                sub_pos, False, spawn_state.difficulty, slot_idx, step_counter
            )
            survived_value = jnp.where(
                survived, direction_of_sub, new_survived[slot_idx]
            ).astype(survived_dtype)

            return new_pos, survived_value

        # Process all subs in this lane simultaneously
        new_positions, survived_values = jax.vmap(process_sub)(
            lane_subs, slot_indices
        )

        # Update positions using dynamic_update_slice
        new_sub_positions = jax.lax.dynamic_update_slice(
            new_sub_positions, new_positions, (base_idx, 0)
        )

        # For survived values
        new_survived = jax.lax.dynamic_update_slice(
            new_survived, survived_values, (base_idx,)
        )

        # Get survived status for lane
        lane_survived = survived_values  # We already have this from above

        # Randomize lane direction if any sub survived
        direction_rng, next_rng = jax.random.split(direction_rng)
        new_lane_directions = jnp.where(
            jnp.any(lane_survived != 0),
            new_lane_directions.at[lane_idx].set(
                jnp.where(
                    jax.random.bernoulli(direction_rng, 0.5),
                    jnp.array(1, dtype=survived_dtype),
                    jnp.array(-1, dtype=survived_dtype)
                )
            ),
            new_lane_directions
        )

        # Direction flipping logic
        new_lane_survived = jnp.where(
            direction_of_sub == -1,
            jnp.flip(lane_survived),
            lane_survived
        )

        # Update lane survived status
        new_survived = jax.lax.dynamic_update_slice(
            new_survived, new_lane_survived, (base_idx,)
        )

        # Get survived status for lane
        lane_survived = survived_values  # We already have this from above

        # Check if any submarine in the lane survived in THIS frame
        # Compare new survived values with old ones to detect changes
        old_survived = jax.lax.dynamic_slice(spawn_state.survived, (base_idx,), (3,))
        any_new_survived = jnp.any(jnp.logical_and(lane_survived != 0, old_survived == 0))

        # Update diver_array if needed
        new_diver_array = jnp.where(
            jnp.logical_and(any_new_survived, new_diver_array[lane_idx] == -1),
            new_diver_array.at[lane_idx].set(1),
            new_diver_array
        )

        # Reset spawn timer only when an enemy survives in this frame
        new_spawn_timers = new_spawn_timers.at[lane_idx].set(
            jnp.where(
                any_new_survived,
                200,  # Reset to 200 when an enemy survives in this frame
                new_spawn_timers[lane_idx]
            )
        )

        return (new_sub_positions, new_survived, new_lane_directions, new_diver_array, new_spawn_timers, next_rng), None

    # Replace shark lane for-loop with scan
    lane_indices = jnp.arange(4)
    (new_shark_positions, new_survived, new_lane_directions, new_diver_array, new_spawn_timers, direction_rng), _ = jax.lax.scan(
        process_shark_lane,
        (new_shark_positions, new_survived, new_lane_directions, spawn_state.diver_array, spawn_state.spawn_timers, direction_rng),
        lane_indices
    )

    # Replace submarine lane for-loop with scan
    (new_sub_positions, new_survived, new_lane_directions, new_diver_array, new_spawn_timers, direction_rng), _ = jax.lax.scan(
        process_sub_lane,
        (new_sub_positions, new_survived, new_lane_directions, new_diver_array, new_spawn_timers, direction_rng),
        lane_indices
    )

    # Update spawn state with new survived status
    new_spawn_state = spawn_state._replace(
        survived=new_survived,
        lane_directions=new_lane_directions,
        diver_array=new_diver_array,
        spawn_timers=new_spawn_timers
    )

    return new_shark_positions, new_sub_positions, new_spawn_state, direction_rng

@jax.jit
def spawn_divers(
    spawn_state: SpawnState,
    diver_positions: chex.Array,
    shark_positions: chex.Array,
    sub_positions: chex.Array,
    step_counter: chex.Array,
) -> tuple[chex.Array, SpawnState]:
    """Spawn divers according to pattern that depends on collection state.

    Follows these rules:
    1. Divers can only spawn in lanes marked as 'spawnable' in diver_array (value 1)
    2. Divers only spawn in empty lanes (no enemies present)
    3. Divers don't spawn in lanes where submarines will spawn next

    Args:
        spawn_state: Current spawn state containing diver_array
        diver_positions: Current diver positions
        shark_positions: Current shark positions
        sub_positions: Current sub positions
        step_counter: Current step counter

    Returns:
        Updated diver positions and updated spawn state
    """

    def spawn_diver(i, carry):
        # Unpack carry - (positions_array, diver_array)
        positions_array, diver_array = carry

        # Get current diver position
        diver_pos = positions_array[i]

        # Check if a diver exists in this slot
        diver_exists = diver_pos[2] != 0

        # Base index for this lane's enemies
        base_idx = i * 3

        # Check if lane has any active enemies
        lane_empty = jnp.all(
            jnp.array(
                [
                    jnp.logical_and(
                        is_slot_empty(shark_positions[base_idx + j]),
                        is_slot_empty(sub_positions[base_idx + j]),
                    )
                    for j in range(3)
                ]
            )
        )

        # Check if this lane is marked as available for spawning (1 means spawn allowed)
        lane_should_spawn = diver_array[i] == 1

        # Combine spawn conditions
        should_spawn = jnp.logical_and(
            jnp.logical_not(diver_exists),  # No existing diver
            jnp.logical_and(lane_should_spawn, lane_empty),
        )

        # If previous wasn't a sub and something survived, next must be a sub
        next_entity_is_sub = jnp.logical_and(
            jnp.logical_not(spawn_state.prev_sub[i]),  # Previous wasn't a sub
            jnp.any(spawn_state.survived[jnp.array([base_idx, base_idx + 1, base_idx + 2])])  # Something survived
        )

        # Override the check if previous was a submarine (next will be a shark)
        next_entity_is_sub = jnp.where(
            spawn_state.prev_sub[i],  # If previous was a sub
            False,  # Force to False (allow spawning)
            next_entity_is_sub  # Otherwise keep original check
        )

        should_spawn = jnp.logical_and(
            should_spawn, jnp.logical_not(next_entity_is_sub)
        )

        # Determine direction based on the to_be_spawned array
        # Get the relevant to_be_spawned values for this lane
        lane_to_be_spawned = jax.lax.dynamic_slice(
            spawn_state.to_be_spawned, (base_idx,), (3,)
        )

        lane_pattern = spawn_state.lane_dependent_pattern[i]

        # For first wave, use the predefined pattern
        # For other waves, check the to_be_spawned array for direction info
        moving_left = spawn_state.lane_directions[i] == 1

        # Set spawn position and direction
        x_pos = jnp.where(moving_left, 168, 0)
        direction = jnp.where(moving_left, -1, 1)

        # Spawn diver if conditions are met
        new_diver = jnp.where(
            should_spawn,
            jnp.array([x_pos, DIVER_SPAWN_POSITIONS[i], direction]),
            diver_pos,
        )

        # Update the full positions array
        new_positions_array = positions_array.at[i].set(new_diver)

        # Check for lane ready for next spawn cycle
        spawn_next_cycle = jnp.logical_and(diver_array[i] == -1, lane_empty)

        # Then apply the next cycle updates to this temporary array
        final_new_diver_array = diver_array.at[i].set(
            jnp.where(
                spawn_next_cycle,
                jnp.array(1, dtype=jnp.int32),
                diver_array[i],  # Use direct indexing to get the value
            )
        )

        # Return the updated full arrays
        return new_positions_array, final_new_diver_array

    # Helper function to process a lane only if timer is at 60
    def process_lane_if_ready(i, carry):
        positions, diver_array = carry

        # Check if timer for this lane is exactly 60
        timer_is_60 = spawn_state.spawn_timers[i] == 60

        # Run spawn logic only if timer is 60
        return jax.lax.cond(
            timer_is_60,
            lambda c: spawn_diver(i, c),  # Process the lane if timer is 60
            lambda c: c,  # Skip processing if timer isn't 60
            (positions, diver_array),
        )

    # Process all lanes, but only execute spawn logic for lanes with timer = 60
    new_diver_positions, new_diver_array = jax.lax.fori_loop(
        0,
        diver_positions.shape[0],
        process_lane_if_ready,
        (diver_positions, spawn_state.diver_array),
    )

    return new_diver_positions, spawn_state._replace(diver_array=new_diver_array)

@jax.jit
def step_diver_movement(
    diver_positions: chex.Array,
    shark_positions: chex.Array,
    state_player_x: chex.Array,
    state_player_y: chex.Array,
    state_divers_collected: chex.Array,
    spawn_state: SpawnState,
    step_counter: chex.Array,
    rng: chex.PRNGKey,
) -> tuple[chex.Array, chex.Array, SpawnState, chex.PRNGKey]:
    """Move divers according to their pattern and handle collisions.
    Returns updated diver positions, number of collected divers, updated spawn state, and updated RNG key.
    """
    new_diver_array = spawn_state.diver_array

    def calculate_diver_movement(step_counter, difficulty):
        """Calculate diver movement based on difficulty level.

        Args:
            step_counter: Current step counter (frame number)
            difficulty: Current difficulty level (0-255)

        Returns:
            Movement speed for the current frame (0, 1, or 2+)
            0 = no movement, 1 = normal speed, 2+ = higher speeds
        """
        # Ensure difficulty is non-negative and handle wrapping
        safe_difficulty = jnp.clip(difficulty % 256, 0, 255)

        # For difficulties 0-27, we have specific movement patterns
        is_high_difficulty = safe_difficulty >= 28

        # For difficulties 0-27, determine if we should move and use speed 1
        low_diff_should_move = determine_low_difficulty_movement(
            step_counter, safe_difficulty
        )
        low_diff_speed = jnp.where(low_diff_should_move, 1, 0)

        # For difficulties 28+, always move but with varying speed
        high_diff_speed = determine_high_difficulty_speed(step_counter, safe_difficulty)

        # Return appropriate speed based on difficulty
        return jnp.where(is_high_difficulty, high_diff_speed, low_diff_speed)

    def determine_low_difficulty_movement(step_counter, difficulty):
        """Determine if the diver should move for difficulties 0-27."""
        # Create boolean masks for each difficulty bracket
        diff_0_1 = jnp.logical_and(difficulty >= 0, difficulty <= 1)
        diff_2_3 = jnp.logical_and(difficulty >= 2, difficulty <= 3)
        diff_4_5 = jnp.logical_and(difficulty >= 4, difficulty <= 5)
        diff_6_7 = jnp.logical_and(difficulty >= 6, difficulty <= 7)
        diff_8_9 = jnp.logical_and(difficulty >= 8, difficulty <= 9)
        diff_10_11 = jnp.logical_and(difficulty >= 10, difficulty <= 11)
        diff_12_13 = jnp.logical_and(difficulty >= 12, difficulty <= 13)
        diff_14_15 = jnp.logical_and(difficulty >= 14, difficulty <= 15)
        diff_16_17 = jnp.logical_and(difficulty >= 16, difficulty <= 17)
        diff_18_19 = jnp.logical_and(difficulty >= 18, difficulty <= 19)
        diff_20_21 = jnp.logical_and(difficulty >= 20, difficulty <= 21)
        diff_22_23 = jnp.logical_and(difficulty >= 22, difficulty <= 23)
        diff_24_25 = jnp.logical_and(difficulty >= 24, difficulty <= 25)
        diff_26_27 = jnp.logical_and(difficulty >= 26, difficulty <= 27)

        # Movement patterns for each bracket based on paste.txt
        # Difficulty 0-1: Move every 5th frame (20% movement)
        move_0_1 = (step_counter % 5) == 0

        # Difficulty 2-3: Move every 4th frame (25% movement)
        move_2_3 = (step_counter % 4) == 0

        # Difficulty 4-5: Move every 3rd frame (33.3% movement)
        move_4_5 = (step_counter % 3) == 0

        # Difficulty 6-7: Move in pattern [1,0,0,1,0,1,0,0] (37.5% movement)
        cycle_6_7 = step_counter % 8
        move_6_7 = jnp.logical_or(
            cycle_6_7 == 0, jnp.logical_or(cycle_6_7 == 3, cycle_6_7 == 5)
        )

        # Difficulty 8-9: Move in pattern [1,0,1,0,1,0,0,1,0,1] (50% movement)
        cycle_8_9 = step_counter % 10
        move_8_9 = jnp.logical_or(
            jnp.logical_or(cycle_8_9 == 0, cycle_8_9 == 2),
            jnp.logical_or(cycle_8_9 == 4, cycle_8_9 == 7),
        )
        move_8_9 = jnp.logical_or(move_8_9, cycle_8_9 == 9)

        # Difficulty 10-11: Move every other frame (50% movement)
        move_10_11 = (step_counter % 2) == 0

        # Difficulty 12-13: Complex pattern with ~60% movement
        cycle_12_13 = step_counter % 8
        move_12_13 = jnp.logical_or(
            jnp.logical_or(cycle_12_13 == 0, cycle_12_13 == 2),
            jnp.logical_or(cycle_12_13 == 4, cycle_12_13 == 6),
        )
        move_12_13 = jnp.logical_or(move_12_13, cycle_12_13 == 7)

        # Difficulty 14-15: Complex pattern with ~65% movement
        cycle_14_15 = step_counter % 7
        move_14_15 = jnp.logical_or(
            jnp.logical_or(cycle_14_15 == 0, cycle_14_15 == 1),
            jnp.logical_or(cycle_14_15 == 3, cycle_14_15 == 5),
        )
        move_14_15 = jnp.logical_or(move_14_15, cycle_14_15 == 6)

        # Difficulty 16-17: Complex pattern with ~70% movement
        cycle_16_17 = step_counter % 10
        move_16_17 = jnp.logical_or(
            jnp.logical_or(cycle_16_17 == 0, cycle_16_17 == 1),
            jnp.logical_or(cycle_16_17 == 3, cycle_16_17 == 4),
        )
        move_16_17 = jnp.logical_or(
            move_16_17, jnp.logical_or(cycle_16_17 == 6, cycle_16_17 == 8)
        )
        move_16_17 = jnp.logical_or(move_16_17, cycle_16_17 == 9)

        # Difficulty 18-19: Move 3 out of 4 frames (75% movement)
        move_18_19 = (step_counter % 4) != 3

        # Difficulty 20-21: Move 4 out of 5 frames (80% movement)
        move_20_21 = (step_counter % 5) != 4

        # Difficulty 22-23: Move 7 out of 8 frames (87.5% movement)
        move_22_23 = (step_counter % 8) != 7

        # Difficulty 24-25: Move 15 out of 16 frames (93.75% movement)
        move_24_25 = (step_counter % 16) != 15

        # Difficulty 26-27: Always move (100% movement)
        move_26_27 = True

        # Combine all patterns using jnp.select which is cleaner for many conditions
        # Create condition array - only the first True condition will be used
        conditions = jnp.array(
            [
                diff_0_1,
                diff_2_3,
                diff_4_5,
                diff_6_7,
                diff_8_9,
                diff_10_11,
                diff_12_13,
                diff_14_15,
                diff_16_17,
                diff_18_19,
                diff_20_21,
                diff_22_23,
                diff_24_25,
                diff_26_27,
            ]
        )

        # Create corresponding values array
        values = jnp.array(
            [
                move_0_1,
                move_2_3,
                move_4_5,
                move_6_7,
                move_8_9,
                move_10_11,
                move_12_13,
                move_14_15,
                move_16_17,
                move_18_19,
                move_20_21,
                move_22_23,
                move_24_25,
                move_26_27,
            ]
        )

        # Select the appropriate pattern based on which condition is True
        should_move = jnp.select(conditions, values, default=False)

        return should_move

    def determine_high_difficulty_speed(step_counter, difficulty):
        """Determine the speed (1 or 2+) for difficulties 28+."""
        # Adjust difficulty to start from 0 for easier tier calculations
        diff_above_27 = difficulty - 28

        # Each 16 difficulty levels form a tier (just like in shark/submarine algorithm)
        tier = diff_above_27 // 16
        position_in_tier = diff_above_27 % 16

        # Base speed for each tier (increases by 1 for each tier)
        base_speed = tier + 1
        higher_speed = tier + 2

        # Position brackets within tier (matches the pattern observed in paste.txt)
        pos_0 = position_in_tier == 0
        pos_1_3 = jnp.logical_and(position_in_tier >= 1, position_in_tier <= 3)
        pos_4_6 = jnp.logical_and(position_in_tier >= 4, position_in_tier <= 6)
        pos_7_9 = jnp.logical_and(position_in_tier >= 7, position_in_tier <= 9)
        pos_10_12 = jnp.logical_and(position_in_tier >= 10, position_in_tier <= 12)
        pos_13_14 = jnp.logical_and(position_in_tier >= 13, position_in_tier <= 14)
        pos_15 = position_in_tier == 15

        # Determine higher speed frequency based on position in tier
        # These frequencies match the observed patterns in paste.txt
        use_higher_speed_pos_0 = (step_counter % 16) == 15  # 1 in 16 frames (6.25%)
        use_higher_speed_pos_1_3 = (step_counter % 8) == 7  # 1 in 8 frames (12.5%)
        use_higher_speed_pos_4_6 = (step_counter % 4) == 3  # 1 in 4 frames (25%)
        use_higher_speed_pos_7_9 = (step_counter % 2) == 1  # 1 in 2 frames (50%)
        use_higher_speed_pos_10_12 = (step_counter % 4) != 0  # 3 in 4 frames (75%)
        use_higher_speed_pos_13_14 = (step_counter % 8) != 0  # 7 in 8 frames (87.5%)
        use_higher_speed_pos_15 = (step_counter % 16) != 0  # 15 in 16 frames (93.75%)

        # Select the appropriate higher speed frequency based on position
        # Use jnp.select for cleaner code with multiple conditions
        position_conditions = jnp.array(
            [pos_0, pos_1_3, pos_4_6, pos_7_9, pos_10_12, pos_13_14, pos_15]
        )

        speed_values = jnp.array(
            [
                use_higher_speed_pos_0,
                use_higher_speed_pos_1_3,
                use_higher_speed_pos_4_6,
                use_higher_speed_pos_7_9,
                use_higher_speed_pos_10_12,
                use_higher_speed_pos_13_14,
                use_higher_speed_pos_15,
            ]
        )

        use_higher_speed = jnp.select(position_conditions, speed_values, default=False)

        # Calculate final speed: higher_speed or base_speed
        return jnp.where(use_higher_speed, higher_speed, base_speed)

    def move_single_diver(i, carry):
        # Unpack carry state - (positions, collected_count, diver_array)
        positions, collected, diver_array = carry
        diver_pos = positions[i]

        # Only process active divers (direction != 0)
        is_active = diver_pos[2] != 0

        # Check for collision with player first if diver is active
        player_collision = jnp.logical_and(
            is_active,
            check_collision_single(
                jnp.array([state_player_x, state_player_y]),
                PLAYER_SIZE,
                jnp.array([diver_pos[0], diver_pos[1]]),
                DIVER_SIZE,
            ),
        )

        # Only collect if we haven't reached max divers
        can_collect = state_divers_collected < 6
        should_collect = jnp.logical_and(player_collision, can_collect)

        # Get the three sharks in the lane
        all_shark_lane_pos = jax.lax.dynamic_slice(shark_positions, (i * 3, 0), (3, 3))

        # Get shark in the same lane for collision check
        shark_lane_pos = get_front_entity(i, all_shark_lane_pos)
        shark_collision = jnp.logical_and(
            is_active,
            check_collision_single(
                jnp.array([shark_lane_pos[0], shark_lane_pos[1]]),
                SHARK_SIZE,
                jnp.array([diver_pos[0], diver_pos[1]]),
                DIVER_SIZE,
            ),
        )

        # check in which direction the shark is moving and copy the direction to the diver
        direction_of_shark = jnp.where(
            shark_lane_pos[2] == 0, diver_pos[2], shark_lane_pos[2]
        )

        # Calculate movement based on difficulty
        movement_speed = calculate_diver_movement(step_counter, spawn_state.difficulty)
        should_move = movement_speed > 0

        # Calculate movement direction (with speed factor)
        # If colliding with shark, use shark's direction/speed
        # Otherwise use diver's direction with appropriate speed factor
        movement_x = jnp.where(
            shark_collision,
            shark_lane_pos[2],  # Use shark's direction/speed
            diver_pos[2] * movement_speed,  # Apply difficulty-based speed
        )

        # Calculate new position
        new_x = jnp.where(
            shark_collision,
            diver_pos[0] + movement_x,  # Move with shark
            jnp.where(
                should_move,
                diver_pos[0] + movement_x,  # Move with calculated speed
                diver_pos[0],  # Stay still
            ),
        )

        # Check bounds
        out_of_bounds = jnp.logical_or(new_x <= -8, new_x >= 170)

        # Create new position array - handle collection and bounds
        new_pos = jnp.where(
            jnp.logical_or(~is_active, jnp.logical_or(out_of_bounds, should_collect)),
            jnp.zeros(3),  # Reset if out of bounds or collected
            jnp.array([new_x, DIVER_SPAWN_POSITIONS[i], direction_of_shark]),
        )

        # Update collection count if collected
        new_collected = collected + jnp.where(should_collect, 1, 0)

        # Update diver collection tracking - mark lane as collected when diver is collected
        updated_diver_array = diver_array.at[i].set(
            jnp.where(should_collect, 0, diver_array[i])
        )

        # if the diver went out of bounds set the entry to -1
        updated_diver_array = updated_diver_array.at[i].set(
            jnp.where(out_of_bounds, -1, updated_diver_array[i])
        )

        # Update the diver position, collection count and diver_array
        return positions.at[i].set(new_pos), new_collected, updated_diver_array

    # Update all diver positions and track collections
    initial_carry = (diver_positions, state_divers_collected, new_diver_array)
    final_positions, final_collected, final_diver_array = jax.lax.fori_loop(
        0, diver_positions.shape[0], move_single_diver, initial_carry
    )

    # Handle case where all divers are collected - set all lanes to -1
    # Apply the reset only if all divers have been collected
    reset_array = jnp.where(
        jnp.all(final_diver_array == 0),
        jnp.array([-1, -1, -1, -1], dtype=jnp.int32),  # Randomized reset array
        final_diver_array,  # Otherwise keep current state
    )

    # Create updated spawn state
    updated_spawn_state = spawn_state._replace(diver_array=reset_array)

    return final_positions, final_collected, updated_spawn_state, rng

@jax.jit
def spawn_step(
    state,
    spawn_state: SpawnState,
    shark_positions: chex.Array,
    sub_positions: chex.Array,
    diver_positions: chex.Array,
    rng_key: chex.PRNGKey,
) -> Tuple[SpawnState, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Main spawn handling function to be called in game step"""
    # Move existing enemies
    new_shark_positions, new_sub_positions, spawn_state_after_movement, new_key = (
        step_enemy_movement(
            spawn_state, shark_positions, sub_positions, state.step_counter, rng_key
        )
    )

    # Update spawns using updated spawn state
    new_spawn_state, new_shark_positions, new_sub_positions, new_key = (
        update_enemy_spawns(
            spawn_state_after_movement,
            new_shark_positions,
            new_sub_positions,
            diver_positions,
            state.step_counter,
            new_key,
        )
    )

    # Spawn new divers with updated tracking
    new_diver_positions, final_spawn_state = spawn_divers(
        new_spawn_state,
        diver_positions,
        new_shark_positions,
        new_sub_positions,
        state.step_counter,
    )

    return (
        final_spawn_state,
        new_shark_positions,
        new_sub_positions,
        new_diver_positions,
        new_key,
    )


def surface_sub_step(state: SeaquestState) -> chex.Array:
    # Check direction value specifically to get scalar boolean
    sub_exists = state.surface_sub_position[2] != 0

    def spawn_sub(_):
        return jnp.array([159, 45, -1])  # Always spawns right facing left

    def move_sub(carry):
        sub_pos = carry
        new_x = jnp.where(
            state.step_counter % 4 == 0,
            sub_pos[0] - 1,  # Direction always -1
            sub_pos[0],
        )

        # Return either zeros or new position
        return jnp.where(
            jnp.logical_or(new_x < -8, sub_pos[2] == 0),
            jnp.zeros(3),
            jnp.array([new_x, 45, -1]),
        )

    # Each condition needs to be scalar
    enough_rescues = state.successful_rescues >= 2
    enough_divers = state.divers_collected >= 1
    correct_timing = jnp.logical_and(
        state.step_counter % 256 == 0, state.step_counter != 0
    )

    # check if the submarine should spawn
    should_spawn = jnp.logical_and(
        jnp.logical_and(enough_rescues, enough_divers),
        jnp.logical_and(correct_timing, ~sub_exists),
    )

    temp1 = spawn_sub(state.surface_sub_position)
    temp2 = move_sub(state.surface_sub_position)

    return jnp.where(should_spawn, temp1, temp2)

@jax.jit
def enemy_missiles_step(
    curr_sub_positions, curr_enemy_missile_positions, step_counter, difficulty
) -> chex.Array:

    def calculate_missile_speed(step_counter, difficulty):
        """JAX-compatible missile speed calculation function"""
        # Base tier size is 16 difficulty levels
        tier_size = 16

        # Determine base speed (1, 2, 3, etc.) based on difficulty tier
        base_speed = 1 + (difficulty // tier_size)

        # Calculate position within the current tier (0-15)
        position_in_tier = difficulty % tier_size

        # Special case for difficulty 0
        is_diff_0 = difficulty == 0

        # Create position bracket array for each pattern
        pos_brackets = jnp.array(
            [
                jnp.logical_and(
                    position_in_tier >= 0, position_in_tier <= 2
                ),  # 0-2: 6.25%
                jnp.logical_and(
                    position_in_tier >= 3, position_in_tier <= 4
                ),  # 3-4: 12.5%
                jnp.logical_and(
                    position_in_tier >= 5, position_in_tier <= 6
                ),  # 5-6: 25%
                jnp.logical_and(
                    position_in_tier >= 7, position_in_tier <= 8
                ),  # 7-8: 50%
                jnp.logical_and(
                    position_in_tier >= 9, position_in_tier <= 10
                ),  # 9-10: 75%
                jnp.logical_and(
                    position_in_tier >= 11, position_in_tier <= 12
                ),  # 11-12: 87.5%
                jnp.logical_and(
                    position_in_tier >= 13, position_in_tier <= 14
                ),  # 13-14: 93.75%
                position_in_tier == 15,  # 15: 100%
            ]
        )

        # Create array of higher speed patterns
        higher_speed_patterns = jnp.array(
            [
                (step_counter % 16) == 0,  # 6.25%
                (step_counter % 8) == 0,  # 12.5%
                (step_counter % 4) == 0,  # 25%
                (step_counter % 2) == 0,  # 50%
                (step_counter % 4) != 0,  # 75%
                (step_counter % 8) != 0,  # 87.5%
                (step_counter % 16) != 0,  # 93.75%
                True,  # 100%
            ]
        )

        # Use jnp.select to choose the pattern
        use_higher_speed = jnp.select(
            pos_brackets, higher_speed_patterns, default=False
        )

        # Higher speed is base_speed + 1
        higher_speed = base_speed + 1

        # Handle difficulty 0 special case
        return jnp.where(
            is_diff_0, 1, jnp.where(use_higher_speed, higher_speed, base_speed)
        )

    def single_missile_step(i, carry):
        # Input i is the loop index, carry is the full array of missile positions
        # Get current submarine and missile for this index
        missile_pos = carry[i]

        # get the current range of the submarines in the lane
        range_start = i * 3
        current_sub_idx = jnp.array([range_start, range_start + 1, range_start + 2])
        lane_subs = curr_sub_positions[current_sub_idx]

        # get the position of the front submarine (thats the only relevant one)
        sub_pos = get_front_entity(i, lane_subs)

        # check if the missile is in frame
        missile_exists = missile_pos[2] != 0

        # check if the missile should be spawned
        should_spawn = jnp.logical_and(
            jnp.logical_not(missile_exists),
            jnp.logical_and(
                sub_pos[0] >= MISSILE_SPAWN_POSITIONS[0],
                sub_pos[0] <= MISSILE_SPAWN_POSITIONS[1],
            ),
        )

        # Calculate new missile position ( x -/+ 4 (depending on direction), y = 47, direction = sub direction)
        new_missile_x = jnp.where(  # could be sub_pos[0] + 4 * sub_pos[2] as well, but this is easier to read
            sub_pos[2] == 1, sub_pos[0] + 4, sub_pos[0] - 4
        )

        new_missile = jnp.where(
            should_spawn,
            jnp.array(
                [new_missile_x, ENEMY_MISSILE_Y[i], sub_pos[2]]
            ),  # Use submarine's direction
            missile_pos,
        )

        movement_speed = calculate_missile_speed(
            step_counter, difficulty
        )
        velocity = movement_speed * new_missile[2]

        new_missile = jnp.where(
            missile_exists,
            jnp.array([new_missile[0] + velocity, new_missile[1], new_missile[2]]),
            new_missile,
        )

        # Check bounds
        new_missile = jnp.where(
            new_missile[0] < X_BORDERS[0],
            jnp.array([0, 0, 0]),
            jnp.where(new_missile[0] > X_BORDERS[1], jnp.array([0, 0, 0]), new_missile),
        )

        # Update the missile position in the full array
        return carry.at[i].set(new_missile)

    # Update all missile positions maintaining the array shape
    new_missile_positions = jax.lax.fori_loop(
        0, 4, single_missile_step, curr_enemy_missile_positions
    )

    return new_missile_positions

@jax.jit
def player_missile_step(
    state: SeaquestState, curr_player_x, curr_player_y, action: chex.Array
) -> chex.Array:
    # check if the player shot this frame
    fire = jnp.any(
        jnp.array(
            [
                action == Action.FIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNFIRE,
                action == Action.DOWNRIGHTFIRE,
                action == Action.DOWNLEFTFIRE,
                action == Action.RIGHTFIRE,
                action == Action.LEFTFIRE,
                action == Action.UPFIRE,
            ]
        )
    )

    # IMPORTANT: do not change the order of this check, since the missile does not move in its first frame!!
    # also check if there is currently a missile in frame by checking if the player_missile_position is empty
    missile_exists = state.player_missile_position[2] != 0

    # if the player shot and there is no missile in frame, then we can shoot a missile
    # the missile y is the current player y position + 7
    # the missile x is either player x + 3 if facing left or player x + 13 if facing right
    new_missile = jnp.where(
        jnp.logical_and(fire, jnp.logical_not(missile_exists)),
        jnp.where(
            state.player_direction == -1,
            jnp.array([curr_player_x + 3, curr_player_y + 7, -1]),
            jnp.array([curr_player_x + 13, curr_player_y + 7, 1]),
        ),
        state.player_missile_position,
    )

    # if a missile is in frame and exists, we move the missile further in the specified direction (5 per tick), also always put the missile at the current player y position
    new_missile = jnp.where(
        missile_exists,
        jnp.array(
            [new_missile[0] + new_missile[2] * 5, curr_player_y + 7, new_missile[2]]
        ),
        new_missile,
    )

    # check if the new positions are still in bounds
    new_missile = jnp.where(
        new_missile[0] < X_BORDERS[0],
        jnp.array([0, 0, 0]),
        jnp.where(new_missile[0] > X_BORDERS[1], jnp.array([0, 0, 0]), new_missile),
    )

    return new_missile

@jax.jit
def update_oxygen(state, player_x, player_y, player_missile_position):
    """Update oxygen levels and handle surfacing mechanics with proper surfacing detection"""
    PLAYER_BREATHING_Y = [47, 52]  # Range where oxygen neither increases nor decreases

    # Detect actual surfacing moment
    at_surface = player_y == 46
    was_underwater = player_y > 46
    just_surfaced = jnp.logical_and(at_surface, state.just_surfaced == 0)

    # Check player state
    decrease_ox = player_y > PLAYER_BREATHING_Y[1]
    has_divers = state.divers_collected >= 0  # Changed to > 0 instead of >= 0
    has_all_divers = state.divers_collected >= 6
    needs_oxygen = state.oxygen < 64

    # Special handling for initialization state
    in_init_state = state.just_surfaced == -1
    started_diving = player_y > PLAYER_START_Y
    filling_init_oxygen = jnp.logical_and(in_init_state, state.oxygen < 64)

    # Surfacing conditions
    increase_ox = jnp.logical_and(at_surface, needs_oxygen)
    stay_same = jnp.logical_and(
        player_y >= PLAYER_BREATHING_Y[0], player_y <= PLAYER_BREATHING_Y[1]
    )

    # Calculate new divers count before other logic
    new_divers_collected = jnp.where(
        jnp.logical_and(just_surfaced, has_divers),
        jnp.where(in_init_state, state.divers_collected, state.divers_collected - 1),
        state.divers_collected,
    )

    # Handle surfacing without divers - prevent during init
    # Only lose life if we started with no divers
    lose_life = jnp.logical_and(
        jnp.logical_and(just_surfaced, new_divers_collected < 0),
        jnp.logical_not(in_init_state),
    )

    # Handle surfacing with all divers
    should_reset = jnp.logical_and(just_surfaced, has_all_divers)

    # Update surfacing flag with consideration for remaining divers
    new_just_surfaced = jnp.where(
        in_init_state,
        jnp.where(
            jnp.logical_and(started_diving, state.oxygen >= 63),
            jnp.array(0),
            jnp.array(-1),
        ),
        jnp.where(
            was_underwater,
            jnp.array(0),
            jnp.where(at_surface, jnp.array(1), state.just_surfaced),
        ),
    )

    # Handle oxygen changes
    new_oxygen = jnp.where(
        filling_init_oxygen,
        jnp.where(state.step_counter % 2 == 0, state.oxygen + 1, state.oxygen),
        jnp.where(
            decrease_ox,
            jnp.where(state.step_counter % 32 == 0, state.oxygen - 1, state.oxygen),
            state.oxygen,
        ),
    )

    # Important: Base blocking decision on has_divers instead of still_has_divers
    can_refill = jnp.logical_and(increase_ox, has_divers)
    new_oxygen = jnp.where(
        jnp.logical_and(can_refill, jnp.logical_not(in_init_state)),
        jnp.where(
            state.oxygen < 64,
            jnp.where(state.step_counter % 2 == 0, state.oxygen + 1, state.oxygen),
            state.oxygen,
        ),
        new_oxygen,
    )

    # Increase difficulty when reaching max oxygen after surfacing
    old_difficulty = state.spawn_state.difficulty
    reached_max = jnp.logical_and(
        jnp.logical_and(new_oxygen >= 64, state.oxygen < 64),
        jnp.logical_not(in_init_state),
    )
    new_difficulty = jnp.where(reached_max, old_difficulty + 1, old_difficulty)

    new_oxygen = jnp.where(stay_same, state.oxygen, new_oxygen)

    # Use has_divers for blocking decision and combine with oxygen check
    should_block = jnp.logical_and(at_surface, needs_oxygen)

    player_x = jnp.where(should_block, state.player_x, player_x)

    player_y = jnp.where(
        should_block,
        jnp.array(46, dtype=jnp.int32),  # Force to exact surface position
        player_y,
    )

    player_missile_position = jnp.where(
        should_block, jnp.zeros(3), player_missile_position
    )

    # Prevent oxygen depletion during init
    oxygen_depleted = jnp.logical_and(
        new_oxygen <= jnp.array(0), jnp.logical_not(in_init_state)
    )

    return (
        new_oxygen,
        player_x,
        player_y,
        player_missile_position,
        oxygen_depleted,
        lose_life,
        new_divers_collected,
        should_reset,
        new_just_surfaced,
        new_difficulty,
    )

@jax.jit
def player_step(
    state: SeaquestState, action: chex.Array
) -> tuple[chex.Array, chex.Array, chex.Array]:
    # implement all the possible movement directions for the player, the mapping is:
    # anything with left in it, add -1 to the x position
    # anything with right in it, add 1 to the x position
    # anything with up in it, add -1 to the y position
    # anything with down in it, add 1 to the y position
    up = jnp.any(
        jnp.array(
            [
                action == Action.UP,
                action == Action.UPRIGHT,
                action == Action.UPLEFT,
                action == Action.UPFIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.UPLEFTFIRE,
            ]
        )
    )
    down = jnp.any(
        jnp.array(
            [
                action == Action.DOWN,
                action == Action.DOWNRIGHT,
                action == Action.DOWNLEFT,
                action == Action.DOWNFIRE,
                action == Action.DOWNRIGHTFIRE,
                action == Action.DOWNLEFTFIRE,
            ]
        )
    )
    left = jnp.any(
        jnp.array(
            [
                action == Action.LEFT,
                action == Action.UPLEFT,
                action == Action.DOWNLEFT,
                action == Action.LEFTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNLEFTFIRE,
            ]
        )
    )
    right = jnp.any(
        jnp.array(
            [
                action == Action.RIGHT,
                action == Action.UPRIGHT,
                action == Action.DOWNRIGHT,
                action == Action.RIGHTFIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.DOWNRIGHTFIRE,
            ]
        )
    )

    player_x = jnp.where(
        right, state.player_x + 1, jnp.where(left, state.player_x - 1, state.player_x)
    )

    player_y = jnp.where(
        down, state.player_y + 1, jnp.where(up, state.player_y - 1, state.player_y)
    )

    # set the direction according to the movement
    player_direction = jnp.where(right, 1, jnp.where(left, -1, state.player_direction))

    # perform out of bounds checks
    player_x = jnp.where(
        player_x < PLAYER_BOUNDS[0][0],
        PLAYER_BOUNDS[0][0],  # Clamp to min player bound
        jnp.where(
            player_x > PLAYER_BOUNDS[0][1],
            PLAYER_BOUNDS[0][1],  # Clamp to max player bound
            player_x,
        ),
    )

    player_y = jnp.where(
        player_y < PLAYER_BOUNDS[1][0],
        PLAYER_BOUNDS[1][0],
        jnp.where(player_y > PLAYER_BOUNDS[1][1], PLAYER_BOUNDS[1][1], player_y),
    )

    return player_x, player_y, player_direction

@jax.jit
def calculate_kill_points(successful_rescues: chex.Array) -> chex.Array:
    """Calculate the points awarded for killing a shark or submarine. Sharks and submarines are worth 20 points.
    The points are increased by 10 for each successful rescue with a maximum of 90."""
    base_points = 20
    max_points = 90
    additional_points = 10 * successful_rescues
    return jnp.minimum(base_points + additional_points, max_points)


class JaxSeaquest(JaxEnvironment[SeaquestState, SeaquestObservation, SeaquestInfo]):
    def __init__(self, reward_funcs: list[callable] =None):
        super().__init__()
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ]
        self.frame_stack_size = 4
        self.obs_size = 5 + 12 * 5 + 12 * 5 + 4 * 5 + 4 * 5 + 5 + 5 + 4

    def flatten_entity_position(self, entity: EntityPosition) -> jnp.ndarray:
        return jnp.concatenate([entity.x, entity.y, entity.width, entity.height, entity.active])

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: SeaquestObservation) -> jnp.ndarray:
        return jnp.concatenate([
            self.flatten_entity_position(obs.player),
            obs.sharks.flatten(),
            obs.submarines.flatten(),
            obs.divers.flatten(),
            obs.enemy_missiles.flatten(),
            self.flatten_entity_position(obs.surface_submarine),
            self.flatten_entity_position(obs.player_missile),
            obs.collected_divers.flatten(),
            obs.player_score.flatten(),
            obs.lives.flatten(),
            obs.oxygen_level.flatten(),
        ])


    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=None,
            dtype=np.uint8,
        )

    @partial(jax.jit, static_argnums=(0, ))
    def _get_observation(self, state: SeaquestState) -> SeaquestObservation:
        # Create player (already scalar, no need for vectorization)
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(PLAYER_SIZE[0]),
            height=jnp.array(PLAYER_SIZE[1]),
            active=jnp.array(1),  # Player is always active
        )

        # Define a function to convert enemy positions to entity format
        def convert_to_entity(pos, size):
            return jnp.array([
                pos[0],  # x position
                pos[1],  # y position
                size[0],  # width
                size[1],  # height
                pos[2] != 0,  # active flag
            ])

        # Apply conversion to each type of entity using vmap

        # Sharks
        sharks = jax.vmap(lambda pos: convert_to_entity(pos, SHARK_SIZE))(
            state.shark_positions
        )

        # Submarines
        submarines = jax.vmap(lambda pos: convert_to_entity(pos, ENEMY_SUB_SIZE))(
            state.sub_positions
        )

        # Divers
        divers = jax.vmap(lambda pos: convert_to_entity(pos, DIVER_SIZE))(
            state.diver_positions
        )

        # Enemy missiles
        enemy_missiles = jax.vmap(lambda pos: convert_to_entity(pos, MISSILE_SIZE))(
            state.enemy_missile_positions
        )

        # Surface submarine (scalar)
        surface_pos = state.surface_sub_position
        surface_sub = EntityPosition(
            x=surface_pos[0],  # First item of first dimension
            y=surface_pos[1],  # First item of second dimension
            width=jnp.array(ENEMY_SUB_SIZE[0]),
            height=jnp.array(ENEMY_SUB_SIZE[1]),
            active=jnp.array(surface_pos[2] != 0),
        )

        # Player missile (scalar)
        missile_pos = state.player_missile_position
        player_missile = EntityPosition(
            x=missile_pos[0],
            y=missile_pos[1],
            width=jnp.array(MISSILE_SIZE[0]),
            height=jnp.array(MISSILE_SIZE[1]),
            active=jnp.array(missile_pos[2] != 0),
        )

        # Return observation
        return SeaquestObservation(
            player=player,
            sharks=sharks,
            submarines=submarines,
            divers=divers,
            enemy_missiles=enemy_missiles,
            surface_submarine=surface_sub,
            player_missile=player_missile,
            collected_divers=state.divers_collected,
            player_score=state.score,
            lives=state.lives,
            oxygen_level=state.oxygen,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: SeaquestState, all_rewards: jnp.ndarray) -> SeaquestInfo:
        return SeaquestInfo(
            successful_rescues=state.successful_rescues,
            difficulty=state.spawn_state.difficulty,
            step_counter=state.step_counter,
            all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: SeaquestState, state: SeaquestState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: SeaquestState, state: SeaquestState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SeaquestState) -> bool:
        return state.lives < 0

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[SeaquestObservation, SeaquestState]:
        """Initialize game state"""
        reset_state = SeaquestState(
            player_x=jnp.array(PLAYER_START_X),
            player_y=jnp.array(PLAYER_START_Y),
            player_direction=jnp.array(0),
            oxygen=jnp.array(0),  # Full oxygen
            divers_collected=jnp.array(0),
            score=jnp.array(0),
            lives=jnp.array(3),
            spawn_state=initialize_spawn_state(),
            diver_positions=jnp.zeros((MAX_DIVERS, 3)),  # 4 divers
            shark_positions=jnp.zeros((MAX_SHARKS, 3)),
            sub_positions=jnp.zeros((MAX_SUBS, 3)),  # x, y, direction
            enemy_missile_positions=jnp.zeros((MAX_ENEMY_MISSILES, 3)),  # 4 missiles
            surface_sub_position=jnp.zeros(3),  # 1 surface sub
            player_missile_position=jnp.zeros(3),  # x,y,direction
            step_counter=jnp.array(0),
            just_surfaced=jnp.array(-1),
            successful_rescues=jnp.array(0),
            death_counter=jnp.array(0),
            rng_key=key,
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state

    @partial(jax.jit, static_argnums=(0, ))
    def step(
        self, state: SeaquestState, action: chex.Array
    ) -> Tuple[SeaquestObservation, SeaquestState, float, bool, SeaquestInfo]:

        previous_state = state
        _, reset_state = self.reset()

        # First handle death animation if active
        def handle_death_animation():
            # Calculate new positions with frozen X coordinates
            shark_y_positions, _, _, _ = step_enemy_movement(
                state.spawn_state,
                state.shark_positions,
                state.sub_positions,
                state.step_counter,
                state.rng_key,
            )

            # Keep X positions from original state, only update Y
            new_shark_positions = state.shark_positions.at[:, 1].set(
                shark_y_positions[:, 1]
            )
            should_hide_player = state.death_counter <= 45

            # Return either final reset or animation frame
            return jax.lax.cond(
                state.death_counter <= 1,
                lambda _: reset_state._replace(
                    lives=state.lives - 1,
                    score=state.score,
                    successful_rescues=state.successful_rescues,
                    divers_collected=jnp.maximum(state.divers_collected - 1, 0),
                    spawn_state=soft_reset_spawn_state(state.spawn_state),
                    # Lose one diver only at end of animation
                ),
                lambda _: state._replace(
                    death_counter=state.death_counter - 1,
                    shark_positions=new_shark_positions,
                    sub_positions=state.sub_positions,
                    enemy_missile_positions=state.enemy_missile_positions,
                    player_missile_position=jnp.zeros(3),
                    player_x=jnp.where(should_hide_player, -100, state.player_x),
                    step_counter=state.step_counter + 1,
                ),
                operand=None,
            )

        def handle_score_freeze():
            # on scoring, the death counter will be set to -(oxygen * 2 + 16 * 6)
            # thats when we get in here, so duplicate the death animation pattern, but decrease the oxygen until its 0
            # Calculate new positions with frozen X coordinates
            shark_y_positions, _, _, _ = step_enemy_movement(
                state.spawn_state,
                state.shark_positions,
                state.sub_positions,
                state.step_counter,
                state.rng_key,
            )

            # Keep X positions from original state, only update Y
            new_shark_positions = state.shark_positions.at[:, 1].set(
                shark_y_positions[:, 1]
            )

            # calculate the new oxygen
            new_ox = jnp.where(
                state.death_counter % 2 == 0, state.oxygen - 1, state.oxygen
            )

            new_ox = jnp.where(new_ox <= 0, jnp.array(0), state.oxygen)

            # Return either final reset or animation frame
            return jax.lax.cond(
                state.death_counter >= -1,
                lambda _: reset_state._replace(
                    player_x=state.player_x,
                    player_y=state.player_y,
                    player_direction=state.player_direction,
                    score=state.score,
                    lives=state.lives,
                    successful_rescues=state.successful_rescues,
                    divers_collected=jnp.array(0),
                    spawn_state=soft_reset_spawn_state(state.spawn_state),
                    surface_sub_position=state.surface_sub_position,
                    oxygen=jnp.array(0),
                ),
                lambda _: state._replace(
                    death_counter=state.death_counter + 1,
                    shark_positions=new_shark_positions,
                    sub_positions=state.sub_positions,
                    enemy_missile_positions=state.enemy_missile_positions,
                    player_missile_position=jnp.zeros(3),
                    step_counter=state.step_counter + 1,
                    oxygen=new_ox,
                ),
                operand=None,
            )

        # Normal game logic starts here
        def normal_game_step():
            # First check if player should be frozen for oxygen refill
            at_surface = state.player_y == 46
            needs_oxygen = state.oxygen < 64
            should_block = jnp.logical_and(at_surface, needs_oxygen)

            # while player is frozen, keep resetting the spawn counter
            new_spawn_state = jax.lax.cond(
                should_block,
                lambda: state.spawn_state._replace(
                    spawn_timers=jnp.array([80, 80, 80, 120])
                ),
                lambda: state.spawn_state,
            )

            state_updated = state._replace(spawn_state=new_spawn_state)

            # If blocked, force position and disable actions
            player_x = jnp.where(should_block, state.player_x, state.player_x)
            player_y = jnp.where(
                should_block, jnp.array(46, dtype=jnp.int32), state.player_y
            )
            action_mod = jnp.where(should_block, jnp.array(Action.NOOP), action)

            # Now calculate movement using potentially modified positions and action
            next_x, next_y, player_direction = player_step(
                state._replace(player_x=player_x, player_y=player_y), action_mod
            )
            player_missile_position = player_missile_step(
                state, next_x, next_y, action_mod
            )

            # Rest of oxygen handling and game logic
            (
                new_oxygen,
                player_x,
                player_y,
                player_missile_position,
                oxygen_depleted,
                lose_life_surfacing,
                new_divers_collected,
                should_reset,
                new_just_surfaced,
                new_difficulty,
            ) = update_oxygen(state, next_x, next_y, player_missile_position)

            # Update divers collected count from oxygen mechanics
            state_updated = state_updated._replace(
                divers_collected=new_divers_collected
            )

            # update the spawn state with the new difficulty
            new_spawn_state = state_updated.spawn_state._replace(
                difficulty=new_difficulty
            )

            # Check missile collisions
            (
                player_missile_position,
                new_shark_positions,
                new_sub_positions,
                new_score,
                updated_spawn_state,
                new_rng_key,
            ) = check_missile_collisions(
                player_missile_position,
                state_updated.shark_positions,
                state_updated.sub_positions,
                state_updated.score,
                state_updated.successful_rescues,
                new_spawn_state,
                state.rng_key,
            )

            # perform all necessary spawn steps
            (
                new_spawn_state,
                new_shark_positions,
                new_sub_positions,
                new_diver_positions,
                new_rng_key,
            ) = spawn_step(
                state_updated,
                updated_spawn_state,
                new_shark_positions,
                new_sub_positions,
                state.diver_positions,
                new_rng_key,
            )

            new_diver_positions, new_divers_collected, new_spawn_state, new_rng_key = (
                step_diver_movement(
                    new_diver_positions,
                    new_shark_positions,
                    player_x,
                    player_y,
                    state_updated.divers_collected,
                    new_spawn_state,
                    state_updated.step_counter,
                    new_rng_key,
                )
            )

            new_surface_sub_pos = surface_sub_step(state_updated)

            state_updated._replace(surface_sub_position=new_surface_sub_pos)

            # update the enemy missile positions
            new_enemy_missile_positions = enemy_missiles_step(
                new_sub_positions,
                state_updated.enemy_missile_positions,
                state_updated.step_counter,
                state_updated.spawn_state.difficulty,
            )

            # append the surface submarine to the other submarines for the collision check
            # check if the player has collided with any of the enemies
            player_collision, collision_points = check_player_collision(
                player_x,
                player_y,
                new_sub_positions,
                new_shark_positions,
                new_surface_sub_pos,
                state_updated.enemy_missile_positions,
                new_score,
                state_updated.successful_rescues,
            )

            lose_life = jnp.any(
                jnp.array([oxygen_depleted, player_collision, lose_life_surfacing])
            )

            # Start death animation but keep divers intact during animation
            death_animation_state = state_updated._replace(
                score=state.score + collision_points,
                death_counter=jnp.array(90),
                spawn_state=soft_reset_spawn_state(state_updated.spawn_state),
            )

            # Calculate points for rescuing divers. Each diver is worth 50 points.
            # Each successful rescue adds 50 points with a maximum of 1000 points each.
            base_points_per_diver = 50
            max_points_per_diver = 1000
            additional_points_per_rescue = 50 * state.successful_rescues
            points_per_diver = jnp.minimum(
                base_points_per_diver + additional_points_per_rescue,
                max_points_per_diver,
            )
            total_diver_points = points_per_diver * state.divers_collected

            # Calculate bonus points for remaining oxygen
            oxygen_bonus = state.oxygen * 20

            # Calculate total points for successful rescue
            total_rescue_points = total_diver_points + oxygen_bonus

            # TODO: somewhere the oxygen is depleted on surfacing, this currently blocks the slow draining of oxygen (which is not gameplay relevant -> low priority)
            # scoring freeze, 16 ticks per diver i.e. 6 * 16 and also 2 ticks per remaining oxygen (which is drained!)
            # Create the scoring state
            scoring_state = state_updated._replace(
                player_x=player_x,
                player_y=player_y,
                player_direction=player_direction,
                lives=state_updated.lives,
                score=state_updated.score + total_rescue_points,
                successful_rescues=state_updated.successful_rescues + 1,
                spawn_state=soft_reset_spawn_state(state_updated.spawn_state)._replace(
                    difficulty=state_updated.spawn_state.difficulty + 1
                ),
                death_counter=jnp.array(-(96 + state_updated.oxygen * 2)),
            )

            # cap the step counter to 1024
            new_step_counter = jnp.where(
                state_updated.step_counter == 1024,
                jnp.array(0),
                state_updated.step_counter + 1,
            )

            # Create the normal returned state
            normal_returned_state = SeaquestState(
                player_x=player_x,
                player_y=player_y,
                player_direction=player_direction,
                oxygen=new_oxygen,
                divers_collected=new_divers_collected,
                score=new_score,
                lives=state_updated.lives,
                spawn_state=new_spawn_state,
                diver_positions=new_diver_positions,
                shark_positions=new_shark_positions,
                sub_positions=new_sub_positions,
                enemy_missile_positions=new_enemy_missile_positions,
                surface_sub_position=new_surface_sub_pos,
                player_missile_position=player_missile_position,
                step_counter=new_step_counter,
                just_surfaced=new_just_surfaced,
                successful_rescues=state_updated.successful_rescues,
                death_counter=jnp.array(0),
                rng_key=new_rng_key,
            )

            # First handle surfacing with all divers (scoring)
            intermediate_state = jax.lax.cond(
                should_reset,
                lambda _: scoring_state,
                lambda _: normal_returned_state,
                operand=None,
            )

            # Then handle life loss - start death animation instead of immediate reset
            final_state = jax.lax.cond(
                lose_life,
                lambda _: death_animation_state,
                lambda _: intermediate_state,
                operand=None,
            )

            # Check for additional life every 10,000 points
            additional_lives = (final_state.score // 10000) - (state.score // 10000)
            new_lives = jnp.minimum(final_state.lives + additional_lives, 6) # max 6 lives possible

            # Update the final state with new lives
            final_state = final_state._replace(lives=new_lives)

            # Check if the game is over
            game_over = final_state.lives <= -1

            # Handle game over state
            return jax.lax.cond(
                game_over,
                lambda _: reset_state._replace(
                    score=final_state.score,
                    lives=jnp.array(-1),
                    death_counter=jnp.array(0),
                ),
                lambda _: final_state,
                operand=None,
            )

        return_state = jax.lax.cond(
            state.death_counter > 0,
            lambda _: handle_death_animation(),
            lambda _: jax.lax.cond(
                state.death_counter < 0,
                lambda _: handle_score_freeze(),
                lambda _: normal_game_step(),
                operand=None,
            ),
            operand=None,
        )

        # Get observation and info
        observation = self._get_observation(return_state)

        done = self._get_done(return_state)
        env_reward = self._get_env_reward(previous_state, return_state)
        all_rewards = self._get_all_rewards(previous_state, return_state)
        info = self._get_info(return_state, all_rewards)

        # Choose between death animation and normal game step
        return observation, return_state, env_reward, done, info

from jaxatari.renderers import AtraJaxisRenderer

class SeaquestRenderer(AtraJaxisRenderer):
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        # render background
        frame_bg = aj.get_sprite_frame(SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # render player submarine
        frame_pl_sub = aj.get_sprite_frame(SPRITE_PL_SUB, state.step_counter)
        raster = aj.render_at(
            raster,
            state.player_x,
            state.player_y,
            frame_pl_sub,
            flip_horizontal=state.player_direction == FACE_LEFT,
        )

        # render player torpedo
        frame_pl_torp = aj.get_sprite_frame(SPRITE_PL_TORP, state.step_counter)
        should_render = state.player_missile_position[0] > 0
        raster = jax.lax.cond(
            should_render,
            lambda r: aj.render_at(
                r,
                state.player_missile_position[0],
                state.player_missile_position[1],
                frame_pl_torp,
                flip_horizontal=state.player_missile_position[2] == FACE_LEFT,
            ),
            lambda r: r,
            raster,
        )

        # render divers
        frame_diver = aj.get_sprite_frame(SPRITE_DIVER, state.step_counter)
        diver_positions = state.diver_positions

        def render_diver(i, raster_base):
            should_render = diver_positions[i][0] > 0
            return jax.lax.cond(
                should_render,
                lambda r: aj.render_at(
                    r,
                    diver_positions[i][0],
                    diver_positions[i][1],
                    frame_diver,
                    flip_horizontal=(diver_positions[i][2] == FACE_LEFT),
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, MAX_DIVERS, render_diver, raster)

        # render sharks
        frame_shark = aj.get_sprite_frame(SPRITE_SHARK, state.step_counter)

        def render_shark(i, raster_base):
            should_render = state.shark_positions[i][0] > 0
            return jax.lax.cond(
                should_render,
                lambda r: aj.render_at(
                    r,
                    state.shark_positions[i][0],
                    state.shark_positions[i][1],
                    frame_shark,
                    flip_horizontal=(state.shark_positions[i][2] == FACE_LEFT),
                ),
                lambda r: r,
                raster_base,
            )

        # Use fori_loop to render all sharks
        raster = jax.lax.fori_loop(0, MAX_SHARKS, render_shark, raster)

        # render enemy subs
        frame_enemy_sub = aj.get_sprite_frame(SPRITE_ENEMY_SUB, state.step_counter)

        def render_enemy_sub(i, raster_base):
            should_render = state.sub_positions[i][0] > 0
            return jax.lax.cond(
                should_render,
                lambda r: aj.render_at(
                    r,
                    state.sub_positions[i][0],
                    state.sub_positions[i][1],
                    frame_enemy_sub,
                    flip_horizontal=(state.sub_positions[i][2] == FACE_LEFT),
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, MAX_SUBS, render_enemy_sub, raster)

        def render_enemy_surface_sub(i, raster_base):
            should_render = state.surface_sub_position[0] > 0
            return jax.lax.cond(
                should_render,
                lambda r: aj.render_at(
                    r,
                    state.surface_sub_position[0],
                    state.surface_sub_position[1],
                    frame_enemy_sub,
                    flip_horizontal=(state.surface_sub_position[2] == FACE_LEFT),
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(
            0, MAX_SURFACE_SUBS, render_enemy_surface_sub, raster
        )

        frame_enemy_torp = aj.get_sprite_frame(SPRITE_EN_TORP, state.step_counter)

        def render_enemy_torp(i, raster_base):
            should_render = state.enemy_missile_positions[i][0] > 0
            return jax.lax.cond(
                should_render,
                lambda r: aj.render_at(
                    r,
                    state.enemy_missile_positions[i][0],
                    state.enemy_missile_positions[i][1],
                    frame_enemy_torp,
                    flip_horizontal=(state.enemy_missile_positions[i][2] == FACE_LEFT),
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, MAX_ENEMY_MISSILES, render_enemy_torp, raster)

        # show the scores
        score_array = aj.int_to_digits(state.score, max_digits=8)
        # convert the score to a list of digits
        raster = aj.render_label(raster, 10, 10, score_array, DIGITS, spacing=7)
        raster = aj.render_indicator(
            raster, 10, 20, state.lives, LIFE_INDICATOR, spacing=10
        )
        raster = aj.render_indicator(
            raster, 49, 178, state.divers_collected, DIVER_INDICATOR, spacing=10
        )

        raster = aj.render_bar(
            raster, 49, 170, state.oxygen, 64, 63, 5, OXYGEN_BAR_COLOR, (0, 0, 0, 0)
        )

        # Force the first 8 columns (x=0 to x=7) to be black
        bar_width = 8
        # Assuming raster shape is (Height, Width, Channels)
        # Select all rows (:), the first 'bar_width' columns (0:bar_width), and all channels (:)
        raster = raster.at[0:bar_width, :, :].set(0)

        return raster


def get_human_action() -> chex.Array:
    """Get human action from keyboard with support for diagonal movement and combined fire"""
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    # Diagonal movements with fire
    if up and right and fire:
        return jnp.array(Action.UPRIGHTFIRE)
    if up and left and fire:
        return jnp.array(Action.UPLEFTFIRE)
    if down and right and fire:
        return jnp.array(Action.DOWNRIGHTFIRE)
    if down and left and fire:
        return jnp.array(Action.DOWNLEFTFIRE)

    # Cardinal directions with fire
    if up and fire:
        return jnp.array(Action.UPFIRE)
    if down and fire:
        return jnp.array(Action.DOWNFIRE)
    if left and fire:
        return jnp.array(Action.LEFTFIRE)
    if right and fire:
        return jnp.array(Action.RIGHTFIRE)

    # Diagonal movements
    if up and right:
        return jnp.array(Action.UPRIGHT)
    if up and left:
        return jnp.array(Action.UPLEFT)
    if down and right:
        return jnp.array(Action.DOWNRIGHT)
    if down and left:
        return jnp.array(Action.DOWNLEFT)

    # Cardinal directions
    if up:
        return jnp.array(Action.UP)
    if down:
        return jnp.array(Action.DOWN)
    if left:
        return jnp.array(Action.LEFT)
    if right:
        return jnp.array(Action.RIGHT)
    if fire:
        return jnp.array(Action.FIRE)

    return jnp.array(Action.NOOP)


if __name__ == "__main__":
    # Initialize game and renderer
    game = JaxSeaquest()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
    clock = pygame.time.Clock()

    renderer_AtraJaxis = SeaquestRenderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_obs, curr_state = jitted_reset()

    # Game loop with rendering
    running = True
    frame_by_frame = False
    frameskip = 1
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN or (
                event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        curr_obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )
                        print(f"Observations: {curr_obs}")
                        print(f"Reward: {reward}, Done: {done}, Info: {info}")

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                curr_obs, curr_state, reward, done, info = jitted_step(
                    curr_state, action
                )

        # render and update pygame
        raster = renderer_AtraJaxis.render(curr_state)
        aj.update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
        counter += 1
        clock.tick(60)

    pygame.quit()