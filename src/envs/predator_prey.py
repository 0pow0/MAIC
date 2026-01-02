"""
Predator-Prey Environment adapted from IC3Net for MAIC framework.

Each predator agent can observe a vision square around itself.
Predators get reward when they reach the prey location.
Episode ends when all predators reach the prey (in mixed mode).

Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=STAY (if enabled)
"""

import numpy as np
from smac.env.multiagentenv import MultiAgentEnv


class PredatorPreyEnv(MultiAgentEnv):
    def __init__(
        self,
        n_predators=4,
        n_preys=1,
        dim=10,
        vision=2,
        moving_prey=False,
        mode='cooperative',
        no_stay=False,
        timestep_penalty=-0.05,
        prey_reward=1.0,
        episode_limit=50,
        seed=None,
    ):
        """
        Args:
            n_predators: Number of predator agents (controlled)
            n_preys: Number of prey (fixed position by default)
            dim: Grid dimension (dim x dim)
            vision: Vision range for each predator
            moving_prey: Whether prey moves (not implemented)
            mode: 'cooperative' (shared reward) | 'competitive' | 'mixed'
            no_stay: If True, agents cannot stay in place
            timestep_penalty: Penalty per timestep
            prey_reward: Reward for reaching prey
            episode_limit: Maximum episode length
            seed: Random seed
        """
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Environment parameters
        self.n_predators = n_predators
        self.n_preys = n_preys
        self.n_agents = n_predators  # Only predators are controlled agents
        self.dim = dim
        self.dims = (dim, dim)
        self.vision = vision
        self.moving_prey = moving_prey
        self.mode = mode
        self.stay = not no_stay
        self.episode_limit = episode_limit

        # Reward parameters
        self.TIMESTEP_PENALTY = timestep_penalty
        self.PREY_REWARD = prey_reward

        # Action space: UP, RIGHT, DOWN, LEFT, (STAY)
        self.n_actions = 5 if self.stay else 4

        # Grid encoding classes
        self.BASE = dim * dim
        self.OUTSIDE_CLASS = self.BASE + 1
        self.PREY_CLASS = self.BASE + 2
        self.PREDATOR_CLASS = self.BASE + 3

        # Vocabulary size for one-hot encoding
        # grid positions + outside + prey + predator
        self.vocab_size = self.BASE + 1 + 1 + 1

        # Observation shape: flattened vision grid with one-hot encoding
        self.obs_size = self.vocab_size * (2 * self.vision + 1) * (2 * self.vision + 1)

        # State shape: all predator positions + all prey positions (x, y for each)
        self.state_size = (self.n_predators + self.n_preys) * 2

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0

        # Initialize
        self.predator_loc = None
        self.prey_loc = None
        self.grid = None
        self.reached_prey = None
        self.episode_over = False

    def reset(self):
        """Reset environment and return initial observations and state."""
        self._episode_steps = 0
        self.episode_over = False
        self.reached_prey = np.zeros(self.n_predators)

        # Random initial locations (no overlap)
        locs = self._get_coordinates()
        self.predator_loc = locs[:self.n_predators].copy()
        self.prey_loc = locs[self.n_predators:].copy()

        self._set_grid()

        return self.get_obs(), self.get_state()

    def step(self, actions):
        """Execute actions and return reward, terminated, info."""
        if self.episode_over:
            raise RuntimeError("Episode is done")

        self._total_steps += 1
        self._episode_steps += 1

        actions = np.array(actions).flatten()

        # Execute actions for each predator
        for i, a in enumerate(actions):
            self._take_action(i, int(a))

        # Calculate reward
        reward = self._get_reward()

        # Check termination
        terminated = self.episode_over
        if self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

        info = {
            'battle_won': np.all(self.reached_prey == 1),
        }

        if info['battle_won']:
            self.battles_won += 1

        return reward, terminated, info

    def _get_coordinates(self):
        """Get random non-overlapping coordinates for all agents."""
        n_total = self.n_predators + self.n_preys
        idx = np.random.choice(np.prod(self.dims), n_total, replace=False)
        return np.vstack(np.unravel_index(idx, self.dims)).T

    def _set_grid(self):
        """Initialize the grid with padding for vision."""
        self.grid = np.arange(self.BASE).reshape(self.dims)
        # Pad grid for vision (outside boundary marked as OUTSIDE_CLASS)
        self.grid = np.pad(self.grid, self.vision, 'constant',
                          constant_values=self.OUTSIDE_CLASS)
        self.empty_bool_base_grid = self._onehot_initialization(self.grid)

    def _onehot_initialization(self, a):
        """Create one-hot encoding of grid."""
        ncols = self.vocab_size
        out = np.zeros(a.shape + (ncols,), dtype=np.float32)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def _take_action(self, idx, act):
        """Execute action for predator idx."""
        if self.reached_prey[idx] == 1:
            return

        # STAY action
        if act == 4:
            return

        # UP
        if act == 0:
            new_pos = max(0, self.predator_loc[idx][0] - 1)
            self.predator_loc[idx][0] = new_pos
        # RIGHT
        elif act == 1:
            new_pos = min(self.dim - 1, self.predator_loc[idx][1] + 1)
            self.predator_loc[idx][1] = new_pos
        # DOWN
        elif act == 2:
            new_pos = min(self.dim - 1, self.predator_loc[idx][0] + 1)
            self.predator_loc[idx][0] = new_pos
        # LEFT
        elif act == 3:
            new_pos = max(0, self.predator_loc[idx][1] - 1)
            self.predator_loc[idx][1] = new_pos

    def _get_reward(self):
        """Calculate team reward."""
        reward = self.TIMESTEP_PENALTY

        # Check which predators are on prey
        on_prey = []
        for i, pred_loc in enumerate(self.predator_loc):
            for prey_loc in self.prey_loc:
                if np.array_equal(pred_loc, prey_loc):
                    on_prey.append(i)
                    break

        nb_predator_on_prey = len(on_prey)

        if self.mode == 'cooperative':
            # Shared reward when any predator reaches prey
            if nb_predator_on_prey > 0:
                reward += self.PREY_REWARD * nb_predator_on_prey
        elif self.mode == 'mixed':
            # Episode ends when all predators reach prey
            if nb_predator_on_prey > 0:
                reward += self.PREY_REWARD * nb_predator_on_prey

        # Mark predators that reached prey
        for i in on_prey:
            self.reached_prey[i] = 1

        # Episode over when all predators reached prey (mixed mode)
        if self.mode == 'mixed' and np.all(self.reached_prey == 1):
            self.episode_over = True

        return reward

    def get_obs(self):
        """Return observations for all agents."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Return observation for a specific agent."""
        # Create grid with agent positions
        bool_base_grid = self.empty_bool_base_grid.copy()

        # Mark predators
        for i, p in enumerate(self.predator_loc):
            bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS] = 1

        # Mark prey
        for p in self.prey_loc:
            bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREY_CLASS] = 1

        # Extract vision window for this agent
        p = self.predator_loc[agent_id]
        slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
        slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
        obs = bool_base_grid[slice_y, slice_x]

        return obs.flatten()

    def get_obs_size(self):
        """Return observation size."""
        return self.obs_size

    def get_state(self):
        """Return global state (all agent positions)."""
        state = []
        # Normalize positions to [0, 1]
        for p in self.predator_loc:
            state.extend([p[0] / self.dim, p[1] / self.dim])
        for p in self.prey_loc:
            state.extend([p[0] / self.dim, p[1] / self.dim])
        return np.array(state, dtype=np.float32)

    def get_state_size(self):
        """Return state size."""
        return self.state_size

    def get_avail_actions(self):
        """Return available actions for all agents."""
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Return available actions for an agent (all actions always available)."""
        return [1] * self.n_actions

    def get_total_actions(self):
        """Return total number of actions."""
        return self.n_actions

    def get_env_info(self):
        """Return environment info dict."""
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def get_stats(self):
        """Return environment statistics."""
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / max(1, self.battles_game),
        }
        return stats

    def render(self):
        """Render environment (text-based)."""
        grid = np.full(self.dims, '.', dtype=object)

        for i, p in enumerate(self.prey_loc):
            grid[p[0], p[1]] = 'P'

        for i, p in enumerate(self.predator_loc):
            if grid[p[0], p[1]] == 'P':
                grid[p[0], p[1]] = 'X'  # Predator on prey
            else:
                grid[p[0], p[1]] = str(i)

        print("-" * (self.dim * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("-" * (self.dim * 2 + 1))

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            self._seed = seed

    def save_replay(self):
        pass
