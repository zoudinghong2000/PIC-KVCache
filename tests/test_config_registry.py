import gc

import pytest

from cacheblend_vllm.config import CacheBlendConfig
from cacheblend_vllm.registry import clear_for_test, get_model, register_model


class Model:
    pass


def test_config_normalizes_sequences_and_rejects_unknown_fields():
    config = CacheBlendConfig.from_extra_config(
        {
            "check_layers": 2,
            "recompute_ratios": 0.25,
            "strict_version_check": False,
        }
    )
    assert config.check_layers == (2,)
    assert config.recompute_ratios == (0.25,)
    with pytest.raises(ValueError, match="Unknown"):
        CacheBlendConfig.from_extra_config({"typo": True})


def test_registry_is_engine_scoped_and_does_not_own_model_lifetime():
    clear_for_test()
    model = Model()
    register_model("engine", model)
    assert get_model("engine") is model
    with pytest.raises(RuntimeError, match="different model"):
        register_model("engine", Model())
    del model
    gc.collect()
    with pytest.raises(RuntimeError, match="not registered"):
        get_model("engine")
