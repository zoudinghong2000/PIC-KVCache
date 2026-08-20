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
    assert config.min_saved_tokens == 0
    assert config.max_apc_prefix_to_hit_ratio == 0.0
    assert config.lookup_timeout_ms == 10_000
    assert config.store_workers == 1
    assert config.max_inflight_store_batches == 8
    assert config.selection_strategy == "kv_deviation"
    assert config.query_score_chunk_size == 8
    with pytest.raises(ValueError, match="Unknown"):
        CacheBlendConfig.from_extra_config({"typo": True})
    with pytest.raises(ValueError, match="prefix"):
        CacheBlendConfig.from_extra_config({"max_apc_prefix_to_hit_ratio": -1})
    with pytest.raises(ValueError, match="store_workers"):
        CacheBlendConfig.from_extra_config({"store_workers": 0})
    with pytest.raises(ValueError, match="Unknown"):
        CacheBlendConfig.from_extra_config({"async_fingerprint": True})
    with pytest.raises(ValueError, match="selection_strategy"):
        CacheBlendConfig.from_extra_config({"selection_strategy": "unknown"})
    with pytest.raises(ValueError, match="exactly one"):
        CacheBlendConfig.from_extra_config(
            {"selection_strategy": "sparse_q", "check_layers": [1, 2]}
        )


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
