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


def test_classify_ctrl_layouts():
    from safety_gymnasium.tasks.safe_multi_agent.utils.ma_action_pack import (
        classify_ctrl_layout,
    )

    assert classify_ctrl_layout(
        ["x_0", "z_0", "x_1", "z_1"], num_agents=2, per_agent_dim=2,
    ) == "blocked"
    assert classify_ctrl_layout(
        ["x_0", "x_1", "z_0", "z_1"], num_agents=2, per_agent_dim=2,
    ) == "interleaved"
    assert classify_ctrl_layout(
        ["x", "z", "x1", "z1"], num_agents=2, per_agent_dim=2,
    ) == "blocked"


def test_interleaved_pack_matches_pre_p0_builder():
    """Builder MA pack is interleaved (pre-P0); keep this locked to empirical deploy."""
    agents = ["agent_0", "agent_1"]
    action = {
        "agent_0": np.array([0.1, -0.2], dtype=np.float64),
        "agent_1": np.array([0.3, 0.4], dtype=np.float64),
    }
    global_action = np.stack(
        [action[a] for a in agents], axis=1,
    ).flatten()
    np.testing.assert_allclose(global_action, [0.1, 0.3, -0.2, 0.4])


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
        # MASAR1 gremlin channel is ~empty (self excluded); MASAR2 is live other-agent.
        gremlin_keys = [k for k in flatten_keys if 'gremlins' in k and 'lidar' in k]
        for key in gremlin_keys:
            train_norm = float(np.linalg.norm(np.ravel(train_obs[key])))
            assert train_norm < 1e-5, f'MASAR1 {key!r} should be ~0, got norm={train_norm}'
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
    assert normalized == ["all_entrapped", "any_surface", "entrapped_0", "surface_1"]


def test_normalize_active_propositions_zero_or_one_per_agent():
    from specbench.envs.zones.sar_propositions import normalize_active_propositions

    categories = {"surface", "entrapped"}
    raw = ["surface_0", "entrapped_0", "surface_1"]
    normalized = normalize_active_propositions(
        raw, num_agents=2, categories=categories, include_team_props=False,
    )
    assert normalized.count("surface_0") + normalized.count("entrapped_0") == 1
    assert normalized == ["entrapped_0", "surface_1"]


def test_normalize_active_propositions_keeps_walls():
    from specbench.envs.zones.sar_propositions import normalize_active_propositions

    categories = {"surface", "entrapped"}
    raw = ["surface_0", "walls", "all_entrapped"]
    normalized = normalize_active_propositions(
        raw, num_agents=2, categories=categories, include_team_props=True,
    )
    assert normalized == [
        "all_entrapped", "any_surface", "any_walls", "surface_0", "walls",
    ]


def test_normalize_active_propositions_sa_walls_only():
    """SA (1 agent): emit walls without coordinator any_walls."""
    from specbench.envs.zones.sar_propositions import normalize_active_propositions

    categories = {"surface", "entrapped"}
    raw = ["surface_0", "walls"]
    normalized = normalize_active_propositions(
        raw, num_agents=1, categories=categories, include_team_props=False,
    )
    assert normalized == ["surface_0", "walls"]
    assert "any_walls" not in normalized


def test_normalize_active_propositions_any_walls_alias():
    from specbench.envs.zones.sar_propositions import normalize_active_propositions

    categories = {"surface", "entrapped"}
    raw = ["any_walls", "surface_0"]
    normalized = normalize_active_propositions(
        raw, num_agents=2, categories=categories, include_team_props=True,
    )
    assert "walls" in normalized and "any_walls" in normalized
    assert "any_surface" in normalized


def test_normalize_active_propositions_any_walls_alias_sa_strips_coordinator():
    from specbench.envs.zones.sar_propositions import normalize_active_propositions

    categories = {"surface", "entrapped"}
    raw = ["any_walls", "surface_0"]
    normalized = normalize_active_propositions(
        raw, num_agents=1, categories=categories, include_team_props=False,
    )
    assert normalized == ["surface_0", "walls"]
    assert "any_walls" not in normalized


def test_resolve_any_surface_lidar_key():
    from specbench.envs.zones.sar_propositions import resolve_casualty_lidar_key

    assert resolve_casualty_lidar_key("any_surface", 1) == "surface_casualtys_lidar_1"
    assert resolve_casualty_lidar_key("any_walls", 0) == "walls_lidar_0"

def test_normalize_team_props_at_most_one():
    from specbench.envs.zones.sar_propositions import normalize_team_props

    assert normalize_team_props(["all_entrapped", "all_surface"]) == ["all_entrapped"]
    assert normalize_team_props(["all_surface"]) == ["all_surface"]


def test_walls_in_get_propositions():
    env = make_env('PointLTL0MASAR2WC-v0', flat=False)
    try:
        props = env.get_propositions()
        assert 'walls' in props
        assert 'any_walls' in props
        assert 'any_surface' in props
    finally:
        env.close()


def test_sa_get_propositions_no_any_walls():
    """Single-agent SAR alphabet: walls + per-agent casualties, no any_walls."""
    env = make_env('PointLTL1MASAR1WC-v0', flat=True)
    try:
        props = env.get_propositions()
        assert 'walls' in props
        assert 'any_walls' not in props
        assert 'entrapped_0' in props
        assert 'surface_0' in props
    finally:
        env.close()


def test_gremlin_stays_centered_on_agent_after_turns():
    """Gremlin mocap must track agent XY without heading-orbit drift (L1 WC)."""
    env = make_env('PointLTL1MASAR1WC-v0', flat=True)
    try:
        env.reset(seed=0)
        task = env.unwrapped.task
        assert task.gremlins.num >= 1
        # Prefer yaw-heavy actions so orbit drift would show if offset returned.
        for _ in range(40):
            action = np.array([0.2, 0.9], dtype=np.float32)
            env.step(action)
        agent_xy = task.agent.get_agent_pos(0)[:2]
        gremlin_xy = np.asarray(task.gremlins.pos[0][:2], dtype=float)
        assert np.linalg.norm(agent_xy - gremlin_xy) < 5e-3, (
            f'gremlin drifted: agent={agent_xy} gremlin={gremlin_xy}'
        )
    finally:
        env.close()


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
            walls = 0
            for prop in props:
                if prop == 'walls':
                    walls += 1
                elif prop in ('all_entrapped', 'all_surface'):
                    team.append(prop)
                else:
                    idx = int(prop.rsplit('_', 1)[1])
                    per_agent.setdefault(idx, []).append(prop)
            assert all(len(v) <= 1 for v in per_agent.values())
            assert len(team) <= 1
            assert walls <= 1
            true_props = tuple(sorted(p for p in props))
            assert true_props in valid or true_props == ()
    finally:
        env.close()


def test_closest_point_on_box_xy_outside_and_inside():
    """Oriented-box closest point uses surface, including nearest face when inside."""
    from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (
        closest_point_on_box_xy,
    )

    center = np.array([0.0, 0.0])
    # Axis-aligned: yaw=0, half extents (1, 2)
    np.testing.assert_allclose(
        closest_point_on_box_xy([5.0, 0.0], center, 0.0, 1.0, 2.0),
        [1.0, 0.0],
    )
    np.testing.assert_allclose(
        closest_point_on_box_xy([0.0, 5.0], center, 0.0, 1.0, 2.0),
        [0.0, 2.0],
    )
    # Outside near a long face, away from center along Y → surface at x=1, y=1.5
    np.testing.assert_allclose(
        closest_point_on_box_xy([3.0, 1.5], center, 0.0, 1.0, 2.0),
        [1.0, 1.5],
    )
    # Inside → nearest face (closer to +X than to ±Y)
    np.testing.assert_allclose(
        closest_point_on_box_xy([0.8, 0.0], center, 0.0, 1.0, 2.0),
        [1.0, 0.0],
    )
    # 90° yaw swaps local axes in world
    np.testing.assert_allclose(
        closest_point_on_box_xy([0.0, 5.0], center, np.pi / 2, 1.0, 2.0),
        [0.0, 1.0],
        atol=1e-9,
    )


def test_walls_lidar_uses_closest_surface_not_center():
    """walls_lidar targets nearest wall surface point, not body centers."""
    from unittest.mock import patch

    env = make_env('PointLTL1MASAR1-v0', flat=True)
    try:
        env.reset(seed=0)
        task = env.unwrapped.task
        walls = task.walls
        assert walls.num > 0
        assert hasattr(walls, 'closest_surface_pos')

        # Surface targets must keep body Z so occluded LOS rays hit elevated boxes.
        surf0 = walls.closest_surface_pos(0, 0)
        assert surf0.shape == (3,)
        np.testing.assert_allclose(surf0[2], walls.pos[0][2])
        n_los = sum(
            1
            for r in range(walls.num)
            if task._lidar_line_of_sight(0, walls.closest_surface_pos(0, r), walls, r)
        )
        assert n_los > 0

        obs = task.obs()
        interior_lidar = task._obs_lidar_pseudo_occluded_new(0, walls)
        arena = task._arena_ltl_walls()
        assert arena is not None
        arena_lidar = task._arena_walls_lidar(0)
        expected = np.maximum(interior_lidar, arena_lidar)
        np.testing.assert_allclose(obs['walls_lidar_0'], expected)
        assert float(expected.max()) > 0.0
        # Arena merge can only raise (or leave) bins vs interior-only.
        assert float(obs['walls_lidar_0'].max()) >= float(interior_lidar.max()) - 1e-9

        with patch.object(
            walls,
            'closest_surface_pos',
            side_effect=lambda agent_idx, row: walls.pos[row],
        ):
            center_lidar = task._obs_lidar_pseudo_occluded_new(0, walls)
        assert not np.allclose(interior_lidar, center_lidar)

        # Agent beside long face far from center: surface closer than body center.
        row = 0
        center = np.asarray(walls.pos[row], dtype=float)[:2]
        size = walls.engine.model.geom(f'wall{row}').size
        xmat = np.asarray(walls.engine.data.body(f'wall{row}').xmat, dtype=float).reshape(3, 3)
        yaw = float(np.arctan2(xmat[1, 0], xmat[0, 0]))
        hx, hy = float(size[0]), float(size[1])
        cos_t, sin_t = np.cos(yaw), np.sin(yaw)
        local = np.array([hx + 0.25, hy * 0.75])
        agent_xy = np.array(
            [
                cos_t * local[0] - sin_t * local[1] + center[0],
                sin_t * local[0] + cos_t * local[1] + center[1],
            ]
        )
        from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (
            closest_point_on_box_xy,
        )

        surface = closest_point_on_box_xy(agent_xy, center, yaw, hx, hy)
        assert np.linalg.norm(agent_xy - surface) + 1e-6 < np.linalg.norm(agent_xy - center)
        np.testing.assert_allclose(
            np.linalg.norm(surface - center),
            np.hypot(hx, hy * 0.75),
            atol=1e-6,
        )
    finally:
        env.close()


def _place_agent_near_xy(task, target_xy, *, max_dist: float = 0.12) -> float:
    """Grid-search Point slide qpos so agent XY is within ``max_dist`` of target."""
    import mujoco

    target = np.asarray(target_xy, dtype=float)[:2]
    best = None
    for dx in np.linspace(-5.0, 5.0, 81):
        for dy in np.linspace(-5.0, 5.0, 81):
            task.data.qpos[0] = float(dx)
            task.data.qpos[1] = float(dy)
            task.data.qpos[2] = 0.0
            mujoco.mj_forward(task.model, task.data)
            pos = np.asarray(task.agent.get_agent_pos(0), dtype=float)[:2]
            dist = float(np.linalg.norm(pos - target))
            if best is None or dist < best[0]:
                best = (dist, float(dx), float(dy))
            if dist <= max_dist:
                return dist
    assert best is not None
    task.data.qpos[0] = best[1]
    task.data.qpos[1] = best[2]
    task.data.qpos[2] = 0.0
    mujoco.mj_forward(task.model, task.data)
    return float(best[0])


def test_surface_lidar_near_touch_peak_and_rescue_skip_lag():
    """Near surface → lidar ≳0.9; rescue-step lag keeps peak; next snapshot zeros it."""
    env = make_env('PointLTL0MASAR1-v0', flat=True)
    try:
        env.reset(seed=0)
        task = env.unwrapped.task
        surface = task.surface_casualtys
        assert surface is not None and surface.num >= 1

        dist = _place_agent_near_xy(task, surface.pos[0], max_dist=0.12)
        assert dist <= 0.12

        obs = task.obs()
        peak = float(np.asarray(obs['surface_casualtys_lidar_0']).max())
        assert peak >= 0.9, f'expected near-field peak, got {peak} at dist={dist}'
        assert task._lidar_line_of_sight(0, surface.pos[0], surface, 0)

        # Clear-LOS regression: reading ≈ geometric expected.
        exp_gain = float(task.lidar_conf.exp_gain)
        expected = float(np.exp(-exp_gain * dist))
        assert abs(peak - expected) < 0.08

        surface.rescued[0] = True
        task._sync_rescued_casualty_state()
        obs_lag = task.obs()
        lag_peak = float(np.asarray(obs_lag['surface_casualtys_lidar_0']).max())
        assert lag_peak >= 0.9, f'rescue-step lidar should still peak, got {lag_peak}'

        task.snapshot_casualty_lidar_skip()
        task._sync_entered_building_state()
        obs_skip = task.obs()
        assert float(np.asarray(obs_skip['surface_casualtys_lidar_0']).max()) == 0.0
        assert 0 not in task._surface_last_seen.get(0, {})
    finally:
        env.close()


def test_surface_lidar_sticky_last_seen_after_occlusion():
    """After first LOS, occluded surface still injects synthetic lidar from last-seen XY."""
    from unittest.mock import patch

    env = make_env('PointLTL0MASAR1-v0', flat=True)
    try:
        env.reset(seed=0)
        task = env.unwrapped.task
        surface = task.surface_casualtys
        _place_agent_near_xy(task, surface.pos[0], max_dist=0.12)

        obs = task.obs()
        assert float(np.asarray(obs['surface_casualtys_lidar_0']).max()) >= 0.9
        assert 0 in task._surface_last_seen[0]

        with patch.object(type(task), '_lidar_line_of_sight', return_value=False):
            # Never-seen agent stays blank.
            task._surface_last_seen[0].clear()
            blank = task.obs()
            assert float(np.asarray(blank['surface_casualtys_lidar_0']).max()) == 0.0
            assert task._surface_sticky_active[0] == set()

            # Restore last-seen → sticky inject.
            task._surface_last_seen[0][0] = np.asarray(surface.pos[0][:2], dtype=float).copy()
            sticky = task.obs()
            sticky_peak = float(np.asarray(sticky['surface_casualtys_lidar_0']).max())
            assert sticky_peak > 0.0
            assert 0 in task._surface_sticky_active[0]
    finally:
        env.close()
