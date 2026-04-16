import importlib.util
import pathlib
import sys

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
    def __init__(self, model):
        self._model = model
        self._input_transform = lambda x: x
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
