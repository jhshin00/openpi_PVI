import importlib.util
import pathlib
import sys

import numpy as np
import pytest


def _load_rtc_module():
    module_path = pathlib.Path(__file__).with_name("main_RTC.py")
    spec = importlib.util.spec_from_file_location("openpi_examples_ur3_main_RTC_test_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RTC module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_rtc = _load_rtc_module()


class _ModelWithDirectHorizon:
    action_horizon = 20

    def realtime_action(self, *args, **kwargs):
        raise NotImplementedError


class _ModelWithConfigHorizon:
    class config:
        action_horizon = 50

    def realtime_action(self, *args, **kwargs):
        raise NotImplementedError


class _Policy:
    def __init__(self, model, *, input_transform=None):
        self._model = model
        self._input_transform = input_transform or (lambda x: x)
        self._output_transform = lambda x: x
        self._sample_kwargs = {}
        self._is_pytorch_model = True


def test_rtc_policy_adapter_reads_direct_action_horizon():
    adapter = _rtc._RTCPolicyAdapter(_Policy(_ModelWithDirectHorizon()))
    assert adapter.supports_rtc is True
    assert adapter.action_horizon == 20


def test_rtc_policy_adapter_reads_config_action_horizon():
    adapter = _rtc._RTCPolicyAdapter(_Policy(_ModelWithConfigHorizon()))
    assert adapter.supports_rtc is True
    assert adapter.action_horizon == 50


def test_validate_rtc_policy_rejects_unknown_horizon():
    class _ModelWithoutHorizon:
        def realtime_action(self, *args, **kwargs):
            raise NotImplementedError

    adapter = _rtc._RTCPolicyAdapter(_Policy(_ModelWithoutHorizon()))
    args = _rtc.Args(policy_dir="unused", replan_steps=8)

    with pytest.raises(ValueError, match="Could not determine the policy action horizon"):
        _rtc._validate_rtc_policy(args, adapter)


def test_prepare_prev_action_chunk_adds_batch_dimension():
    prev_chunk = np.zeros((50, 32), dtype=np.float32)

    prepared = _rtc._RTCPolicyAdapter._prepare_prev_action_chunk(prev_chunk, "cpu")

    assert tuple(prepared.shape) == (1, 50, 32)


def test_prepare_prev_action_chunk_preserves_existing_batch_dimension():
    prev_chunk = np.zeros((1, 50, 32), dtype=np.float32)

    prepared = _rtc._RTCPolicyAdapter._prepare_prev_action_chunk(prev_chunk, "cpu")

    assert tuple(prepared.shape) == (1, 50, 32)


def test_prepare_rtc_prefix_reanchors_absolute_actions_to_current_state():
    state = np.array([0.4, -0.2, 0.1, -0.3, 0.2, -0.1, 1.0], dtype=np.float32)
    absolute_actions = np.array(
        [
            [0.5, -0.1, 0.2, -0.2, 0.4, 0.0, 1.0],
            [0.7, 0.0, 0.3, -0.1, 0.5, 0.1, 0.0],
        ],
        dtype=np.float32,
    )

    def _input_transform(data):
        transformed = dict(data)
        actions = np.asarray(transformed["actions"], dtype=np.float32).copy()
        actions[:, :6] -= np.asarray(transformed["observation/state"], dtype=np.float32)[:6]
        transformed["actions"] = actions
        return transformed

    adapter = _rtc._RTCPolicyAdapter(_Policy(_ModelWithDirectHorizon(), input_transform=_input_transform))
    prepared = adapter.prepare_rtc_prefix(
        {
            "observation/state": state,
            "observation/base_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "prompt": "test",
        },
        absolute_actions,
    )

    expected = absolute_actions.copy()
    expected[:, :6] -= state[:6]
    assert np.allclose(prepared, expected)
