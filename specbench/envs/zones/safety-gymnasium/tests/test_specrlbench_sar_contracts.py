"""SpecRLBench SAR architecture contracts.

These tests pin public behavior before conservative refactors. They are not
intended to prove policy quality; they guard registration, wrapper, reset, and
env integration surfaces that later cleanup must preserve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

SAFETY_ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip('mujoco')

import safety_gymnasium  # noqa: E402
from safety_gymnasium.utils.registration import safe_registry  # noqa: E402
from safety_gymnasium.utils.task_utils import get_task_class_name  # noqa: E402
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (  # noqa: E402
    building_geom,
)

from specbench.envs.zones.zone_env import make_zone_env as make_env  # noqa: E402


SAR_ENV_IDS = {
    'PointLTL0MASAR1-v0': 'MultiGoalSARLevel0',
    'PointLTL1MASAR1-v0': 'MultiGoalSARLevel1',
    'PointLTL2MASAR1-v0': 'MultiGoalSARLevel2',
    'PointLTL0MASAR2-v0': 'MultiGoalSARLevel0',
    'PointLTL1MASAR2-v0': 'MultiGoalSARLevel1',
    'PointLTL2MASAR2-v0': 'MultiGoalSARLevel2',
}


def _layout_snapshot(task) -> tuple[tuple[str, tuple[float, ...]], ...]:
    layout = task.world_info.layout
    return tuple(
        (key, tuple(np.asarray(value, dtype=float).round(8).reshape(-1)))
        for key, value in sorted(layout.items())
    )


def test_sar_env_ids_are_registered_and_resolve_to_expected_tasks():
    """Current public SAR env IDs and class-name mapping are compatibility API."""
    for env_id, expected_class in SAR_ENV_IDS.items():
        assert env_id in safe_registry
        assert get_task_class_name(env_id) == expected_class

        debug_id = env_id.replace('-v0', 'Debug-v0')
        vision_id = env_id.replace('-v0', 'Vision-v0')
        assert debug_id in safe_registry
        assert vision_id in safe_registry
        assert get_task_class_name(debug_id) == expected_class
        assert get_task_class_name(vision_id) == expected_class


def test_sar_non_flat_wrapper_preserves_multi_agent_api():
    """Non-SB3 SAR path returns per-agent dict observations/actions/rewards."""
    env = make_env('PointLTL0MASAR2-v0', flat=False)
    try:
        obs, info = env.reset(seed=0)

        assert set(obs) == {'agent_0', 'agent_1'}
        assert info['propositions'] == []
        assert isinstance(env.observation_space, spaces.Dict)

        action = {
            agent: env.action_space(agent).sample()
            for agent in env.unwrapped.possible_agents
        }
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        assert set(next_obs) == {'agent_0', 'agent_1'}
        assert set(reward) == {'agent_0', 'agent_1'}
        assert set(terminated) == {'agent_0', 'agent_1'}
        assert set(truncated) == {'agent_0', 'agent_1'}
        assert isinstance(step_info['propositions'], list)
        for i, agent in enumerate(env.unwrapped.possible_agents):
            assert f'wall_sensor_{i}' in next_obs[agent]
    finally:
        env.close()


def test_sar_flat_wrapper_flattens_obs_and_actions_for_multiinput_policy():
    """SB3 SAR path exposes one flat Dict obs and one Box action space."""
    env = make_env('PointLTL0MASAR2-v0', flat=True)
    try:
        obs, info = env.reset(seed=0)

        assert isinstance(env.action_space, spaces.Box)
        assert env.action_space.shape == (4,)
        assert isinstance(env.observation_space, spaces.Dict)
        assert set(obs) == set(env.observation_space.spaces)
        assert all(not isinstance(value, dict) for value in obs.values())
        assert info['propositions'] == []

        next_obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())

        assert isinstance(next_obs, dict)
        assert set(next_obs) == set(env.observation_space.spaces)
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(step_info['propositions'], list)
        assert isinstance(step_info['casualty_visible'], bool)
    finally:
        env.close()


@pytest.mark.parametrize('env_id', SAR_ENV_IDS)
def test_all_single_sar_levels_keep_flat_reset_and_step_contract(env_id):
    """Every public SAR level must support SB3 reset and one sampled step."""
    env = make_env(env_id, flat=True)
    try:
        obs, info = env.reset(seed=0)

        assert isinstance(env.action_space, spaces.Box)
        assert env.action_space.shape == (env.unwrapped.num_agents * env.action_dim,)
        assert isinstance(env.observation_space, spaces.Dict)
        assert set(obs) == set(env.observation_space.spaces)
        assert info['propositions'] == []

        next_obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())

        assert set(next_obs) == set(env.observation_space.spaces)
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(step_info['propositions'], list)
    finally:
        env.close()


def test_sar_reset_seed_reproduces_layout_on_same_env():
    """Same explicit reset seed must reproduce authoritative task layout."""
    env = make_env('PointLTL0MASAR1-v0', flat=True)
    try:
        env.reset(seed=7)
        first = _layout_snapshot(env.unwrapped.task)
        env.reset(seed=7)
        second = _layout_snapshot(env.unwrapped.task)
        env.reset(seed=8)
        third = _layout_snapshot(env.unwrapped.task)

        assert first == second
        assert first != third
    finally:
        env.close()


def _building_layout_snapshot(task) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Layout keys produced by building sync (buildings, entrapped, perimeter walls)."""
    layout = task.world_info.layout
    prefixes = ('terracotta_building', 'entrapped_casualty', 'building')
    return tuple(
        (key, tuple(np.asarray(value, dtype=float).round(8).reshape(-1)))
        for key, value in sorted(layout.items())
        if key.startswith(prefixes)
    )


def test_building_entrapped_layout_pinned():
    """Entrapped casualties must spawn at building centers (runtime positions)."""
    env = make_env('PointLTL2MASAR1-v0', flat=True)
    try:
        env.reset(seed=11)
        task = env.unwrapped.task
        assert hasattr(task, 'entrapped_casualtys')
        assert hasattr(task, 'terracotta_buildings')
        entrapped_num = task.entrapped_casualtys.num
        assert entrapped_num > 0

        for i in range(entrapped_num):
            building_xy = np.asarray(task.terracotta_buildings.pos[i][:2], dtype=float)
            entrapped_xy = np.asarray(task.entrapped_casualtys.pos[i][:2], dtype=float)
            np.testing.assert_allclose(building_xy, entrapped_xy, rtol=0, atol=1e-5)
    finally:
        env.close()


def test_building_layout_seed_reproducible():
    """Building layout sync must reproduce on repeated fast-path resets."""
    env = make_env('PointLTL2MASAR1-v0', flat=True)
    try:
        # First reset builds MuJoCo and draws wall sizes; later resets use fast layout resample.
        env.reset(seed=0)
        env.reset(seed=7)
        first = _building_layout_snapshot(env.unwrapped.task)
        env.reset(seed=7)
        second = _building_layout_snapshot(env.unwrapped.task)
        env.reset(seed=8)
        third = _building_layout_snapshot(env.unwrapped.task)

        assert first == second
        assert first != third
    finally:
        env.close()


def test_building_perimeter_wall_keys_exist():
    """Each building must expose four perimeter wall segment layout keys."""
    env = make_env('PointLTL2MASAR2-v0', flat=True)
    try:
        env.reset(seed=3)
        layout = env.unwrapped.task.world_info.layout
        agent_num = env.unwrapped.task.agent_num
        for i in range(agent_num):
            for seg_idx in range(4):
                assert f'building{i}_ltl_wall{seg_idx}' in layout
    finally:
        env.close()


def test_customized_sar_building_num_zero_is_open_field():
    """Explicit building_num=0 must skip buildings, perimeter walls, and casualties."""
    from specbench.envs.zones.safety_gym_register import register_helper

    env_id = 'CustomizedSARBuildingNumZeroTest-v0'
    env_config = {
        'env_id': env_id,
        'agent_name': 'Point',
        'max_episode_steps': 100,
        'agent_num': 1,
        'building_num': 0,
        'wall_count': 4,
        'surface_casualties_per_agent': 2,
        'entrapped_casualties_per_agent': 1,
    }
    register_helper(env_config=env_config)
    env = make_env(env_id, flat=True)
    try:
        env.reset(seed=0)
        task = env.unwrapped.task
        layout = task.world_info.layout
        assert 'terracotta_building0' not in layout
        assert 'building0_ltl_wall0' not in layout
        assert not hasattr(task, 'surface_casualtys')
        assert not hasattr(task, 'entrapped_casualtys')
    finally:
        env.close()


def test_obs_lidar_pseudo_new_empty_positions_is_zeros():
    """Empty building-lidar skip list must not crash (L5 single-building enter)."""
    env = make_env('PointLTL2MASAR1-v0', flat=True)
    try:
        env.reset(seed=11)
        task = env.unwrapped.task
        empty = task._obs_lidar_pseudo_new(0, [])
        assert empty.shape == (task.lidar_conf.num_bins,)
        np.testing.assert_array_equal(empty, np.zeros(task.lidar_conf.num_bins))
    finally:
        env.close()


def test_entered_building_suppresses_shell_lidar_and_render():
    """Entered building shell stays sticky-hidden after exit; visited flag set."""
    from unittest.mock import patch

    from safety_gymnasium.tasks.safe_multi_agent.tasks.single_goal_sar import single_sar_level0

    env = make_env('PointLTL2MASAR1-v0', flat=True)
    try:
        env.reset(seed=11)
        task = env.unwrapped.task
        buildings = building_geom(task)
        assert buildings is not None
        shell_geom_id = task._obstacle_geom_id_for_instance(buildings, 0)
        assert shell_geom_id is not None

        with patch.object(single_sar_level0, 'agent_inside_building_idx', return_value=0):
            task._sync_entered_building_state()
            assert 0 in task._buildings_entered
            assert shell_geom_id in task._lidar_suppressed_geom_ids
            assert task.model.geom_rgba[shell_geom_id][-1] == 0.0

            obs = task.obs()
            positions = [
                buildings.pos[row]
                for row in range(buildings.num)
                if row != 0
            ]
            expected = task._obs_lidar_pseudo_new(0, positions)
            np.testing.assert_array_equal(obs['terracotta_buildings_lidar_0'], expected)
            # L5 has one building: skip sole shell → all-zero building lidar.
            if buildings.num == 1:
                np.testing.assert_array_equal(
                    obs['terracotta_buildings_lidar_0'],
                    np.zeros(task.lidar_conf.num_bins),
                )
            expected_visited = np.zeros(buildings.num, dtype=np.float64)
            expected_visited[0] = 1.0
            np.testing.assert_array_equal(
                obs['terracotta_buildings_visited'],
                expected_visited,
            )

        # Exit: shell stays hidden (sticky for rest of episode).
        with patch.object(single_sar_level0, 'agent_inside_building_idx', return_value=None):
            task._sync_entered_building_state()
            assert 0 in task._buildings_entered
            assert shell_geom_id in task._lidar_suppressed_geom_ids
            assert task.model.geom_rgba[shell_geom_id][-1] == 0.0

            obs = task.obs()
            positions = [
                buildings.pos[row]
                for row in range(buildings.num)
                if row != 0
            ]
            expected = task._obs_lidar_pseudo_new(0, positions)
            np.testing.assert_array_equal(obs['terracotta_buildings_lidar_0'], expected)
            expected_visited = np.zeros(buildings.num, dtype=np.float64)
            expected_visited[0] = 1.0
            np.testing.assert_array_equal(
                obs['terracotta_buildings_visited'],
                expected_visited,
            )
    finally:
        env.close()


def test_wrapper_keeps_entrapped_lidar_when_building_sticky_entered():
    """Entrapped lidar is not force-zeroed once a building is sticky-entered."""
    from unittest.mock import patch

    from safety_gymnasium.tasks.safe_multi_agent.tasks.single_goal_sar import single_sar_level0

    env = make_env('PointLTL2MASAR1-v0', flat=False)
    try:
        env.reset(seed=11)
        task = env.unwrapped.task
        bins = task.lidar_conf.num_bins
        sentinel = np.full(bins, 0.42, dtype=np.float64)

        with patch.object(single_sar_level0, 'agent_inside_building_idx', return_value=0):
            task._sync_entered_building_state()
            assert 0 in task._buildings_entered

        # Outside again, cost pulse gone — sticky entered must still unmask.
        with patch.object(single_sar_level0, 'agent_inside_building_idx', return_value=None):
            task._sync_entered_building_state()
            assert 0 in task._buildings_entered

            fake_obs = {
                'agent_0': {
                    'entrapped_casualtys_lidar_0': sentinel.copy(),
                },
            }
            fake_reward = {'agent_0': 0.0}
            fake_cost = {'agent_0': 0.0}
            fake_terminated = {'agent_0': False}
            fake_truncated = {'agent_0': False}
            fake_info = {
                'agent_0': {
                    'cost_buildings_terracotta': 0.0,
                    'cost_casualtys_surface': 0.0,
                    'cost_casualtys_entrapped': 0.0,
                    'cost_sum': 0.0,
                },
            }

            with patch.object(
                env.env,
                'step',
                return_value=(
                    fake_obs,
                    fake_reward,
                    fake_cost,
                    fake_terminated,
                    fake_truncated,
                    fake_info,
                ),
            ):
                action = {
                    agent: np.zeros(2, dtype=np.float64)
                    for agent in env.unwrapped.possible_agents
                }
                obs, _reward, _terminated, _truncated, _info = env.step(action)

            np.testing.assert_array_equal(
                obs['agent_0']['entrapped_casualtys_lidar_0'],
                sentinel,
            )
    finally:
        env.close()


def test_pointltl_wc_defaults_to_wc_wrapper_not_ltl():
    """GenZ path: PointLTL*WC must not auto-select MASARLTL (LTL substring collision)."""
    from specbench.envs.zones.safety_gym_wrapper_sar_wc import SafetyGymWrapperMASARWC
    from specbench.envs.zones.safety_gym_wrapper_sar_ltl import SafetyGymWrapperMASARLTL

    env = make_env('PointLTL0MASAR1WC-v0', flat=True, sar_ltl_ordering=False)
    try:
        assert isinstance(env, SafetyGymWrapperMASARWC)
        assert not isinstance(env, SafetyGymWrapperMASARLTL)
    finally:
        env.close()


def test_sar_ltl_ordering_opt_in_uses_ltl_wrapper():
    from specbench.envs.zones.safety_gym_wrapper_sar_ltl import SafetyGymWrapperMASARLTL

    env = make_env('PointLTL0MASAR1WC-v0', flat=True, sar_ltl_ordering=True)
    try:
        assert isinstance(env, SafetyGymWrapperMASARLTL)
    finally:
        env.close()


def test_sar_ltl_ordering_surface_rescue_before_entrapped_yields_cost():
    """LTL ordering wrapper penalizes surface rescue before all entrapped are rescued."""
    from unittest.mock import patch

    from specbench.envs.zones.safety_gym_wrapper_sar import SafetyGymWrapperMASAR

    env = make_env('PointLTL0MASAR1WC-v0', flat=True, sar_ltl_ordering=True)
    try:
        env.reset(seed=0)
        fake_obs = env.observation_space.sample()
        fake_reward = 0.0
        fake_terminated = False
        fake_truncated = False
        fake_info = {
            'agent_0': {
                'cost_casualtys_surface': 1.0,
                'cost_walls': 0.0,
                'cost_collision': 0.0,
            },
        }

        with patch.object(SafetyGymWrapperMASAR, 'step', return_value=(
            fake_obs, fake_reward, fake_terminated, fake_truncated, fake_info,
        )):
            _obs, _reward, _terminated, _truncated, info = env.step(env.action_space.sample())

        assert info['cost'] > 0
    finally:
        env.close()


def test_team_props_on_l2_when_all_entrapped_rescued():
    from specbench.envs.zones.sar_propositions import (
        should_expose_team_props,
        team_active_props,
    )

    env = make_env('PointLTL2MASAR2-v0', flat=True)
    try:
        env.reset(seed=11)
        task = env.unwrapped.task
        assert should_expose_team_props(task)
        assert 'all_entrapped' not in team_active_props(task)

        for i in range(task.entrapped_casualtys.num):
            task.entrapped_casualtys.rescued[i] = True

        assert 'all_entrapped' in team_active_props(task)
        assert 'all_entrapped' in env.get_propositions()
    finally:
        env.close()


def test_pointltl0masar2_registers_with_two_agents():
    env = make_env('PointLTL0MASAR2-v0', flat=False)
    try:
        assert env.unwrapped.num_agents == 2
        obs, _ = env.reset(seed=0)
        assert set(obs) == {'agent_0', 'agent_1'}
    finally:
        env.close()


def test_wc_ignores_collision_for_cost_and_termination():
    """Paper protocol: inter-agent collision must not drive WC cost or termination."""
    from unittest.mock import patch

    from specbench.envs.zones.safety_gym_wrapper_sar import SafetyGymWrapperMASAR

    env = make_env('PointLTL0MASAR2WC-v0', flat=True)
    try:
        env.reset(seed=0)
        fake_obs = env.observation_space.sample()
        fake_info = {
            'agent_0': {'cost_collision': 1.0, 'cost_walls': 0.0},
            'agent_1': {'cost_collision': 1.0, 'cost_walls': 0.0},
        }
        with patch.object(SafetyGymWrapperMASAR, 'step', return_value=(
            fake_obs, 0.0, False, False, fake_info,
        )):
            _obs, _reward, terminated, _truncated, info = env.step(env.action_space.sample())

        assert info['cost'] == 0
        assert not terminated
    finally:
        env.close()


def test_ltl_collision_alone_does_not_yield_cost():
    """LTL wrapper ignores collision; entrapped-first is tested separately."""
    from unittest.mock import patch

    from specbench.envs.zones.safety_gym_wrapper_sar import SafetyGymWrapperMASAR

    env = make_env('PointLTL0MASAR1WC-v0', flat=True, sar_ltl_ordering=True)
    try:
        env.reset(seed=0)
        fake_obs = env.observation_space.sample()
        fake_info = {
            'agent_0': {
                'cost_collision': 1.0,
                'cost_walls': 0.0,
                'cost_casualtys_surface': 0.0,
            },
        }
        with patch.object(SafetyGymWrapperMASAR, 'step', return_value=(
            fake_obs, 0.0, False, False, fake_info,
        )):
            _obs, _reward, terminated, _truncated, info = env.step(env.action_space.sample())

        assert info['cost'] == 0
        assert not terminated
    finally:
        env.close()


def _flatten_dict_obs(obs: dict, keys: list[str]) -> np.ndarray:
    parts = [np.ravel(np.asarray(obs[k], dtype=np.float32)) for k in keys]
    return np.concatenate(parts, axis=0).astype(np.float32)


@pytest.mark.parametrize('sar_ltl_ordering', [False, True])
def test_masar1wc_train_obs_matches_masar2wc_agent0_deploy(sar_ltl_ordering: bool):
    """Paper deploy: shared SA policy obs on agent_0 must match MASAR1WC train flat obs."""
    train_env = 'PointLTL0MASAR1WC-v0'
    eval_env = 'PointLTL0MASAR2WC-v0'

    train = make_env(train_env, flat=True, sar_ltl_ordering=sar_ltl_ordering)
    try:
        train_obs, _ = train.reset(seed=7)
        flatten_keys = sorted(train_obs.keys())
    finally:
        train.close()

    eval_ma = make_env(eval_env, flat=False, sar_ltl_ordering=sar_ltl_ordering)
    try:
        eval_obs, _ = eval_ma.reset(seed=7)
        agent0 = eval_obs['agent_0']
        for key in flatten_keys:
            assert key in agent0, f'missing {key!r} on MASAR2 agent_0'
            t_size = np.ravel(train_obs[key]).size
            e_size = np.ravel(agent0[key]).size
            assert t_size == e_size, (
                f'{key!r}: train flat size {t_size} != agent_0 size {e_size}'
            )
        train_flat = _flatten_dict_obs(train_obs, flatten_keys)
        deploy_flat = _flatten_dict_obs(agent0, flatten_keys)
        assert train_flat.shape == deploy_flat.shape
    finally:
        eval_ma.close()


def test_resolve_casualty_lidar_keys_for_observer_falls_back_to_egocentric():
    from specbench.envs.zones.sar_propositions import resolve_casualty_lidar_keys_for_observer

    available = {
        "surface_casualtys_lidar_1",
        "entrapped_casualtys_lidar_1",
    }
    assert resolve_casualty_lidar_keys_for_observer(
        "surface_0", 1, available_keys=available,
    ) == ["surface_casualtys_lidar_1"]
    assert resolve_casualty_lidar_keys_for_observer(
        "all_entrapped", 1, num_agents=2, available_keys=available,
    ) == ["entrapped_casualtys_lidar_1"]


def test_normalize_active_propositions_includes_team_props_when_enabled():
    from specbench.envs.zones.sar_propositions import normalize_active_propositions

    categories = {"surface", "entrapped"}
    raw = ["surface_0", "entrapped_0", "surface_1", "all_entrapped"]
    normalized = normalize_active_propositions(
        raw, num_agents=2, categories=categories, include_team_props=True,
    )
    assert normalized.count("surface_0") + normalized.count("entrapped_0") == 1
    assert normalized == ["all_entrapped", "entrapped_0", "surface_1"]


def test_normalize_active_propositions_zero_or_one_per_agent():
    from specbench.envs.zones.sar_propositions import normalize_active_propositions

    categories = {"surface", "entrapped"}
    raw = ["surface_0", "entrapped_0", "surface_1"]
    normalized = normalize_active_propositions(
        raw, num_agents=2, categories=categories, include_team_props=False,
    )
    assert normalized.count("surface_0") + normalized.count("entrapped_0") == 1
    assert normalized == ["entrapped_0", "surface_1"]


def test_normalize_team_props_at_most_one():
    from specbench.envs.zones.sar_propositions import normalize_team_props

    assert normalize_team_props(["all_entrapped", "all_surface"]) == ["all_entrapped"]
    assert normalize_team_props(["all_surface"]) == ["all_surface"]


def test_step_propositions_are_zero_or_one_per_agent():
    """Runtime props must match get_possible_assignments (Büchi-safe)."""
    env = make_env('PointLTL0MASAR2WC-v0', flat=False)
    try:
        env.reset(seed=3)
        valid = {
            tuple(sorted(a.get_true_propositions()))
            for a in env.get_possible_assignments()
        }
        for _ in range(25):
            action = {
                agent: env.action_space(agent).sample()
                for agent in env.unwrapped.possible_agents
            }
            _obs, _reward, _terminated, _truncated, info = env.step(action)
            props = info['propositions']
            per_agent: dict[int, list[str]] = {}
            team = []
            for prop in props:
                if prop in ('all_entrapped', 'all_surface'):
                    team.append(prop)
                else:
                    idx = int(prop.rsplit('_', 1)[1])
                    per_agent.setdefault(idx, []).append(prop)
            assert all(len(v) <= 1 for v in per_agent.values())
            assert len(team) <= 1
            true_props = tuple(sorted(p for p in props))
            assert true_props in valid or true_props == ()
    finally:
        env.close()
