import pytest
from app.core.config import get_settings
from app.services import fingerspelling_provider, inference_provider


@pytest.fixture
def reset_inference_caches():
    """get_settings() (lru_cache) and inference_provider's module-global
    InferenceService are both process-wide caches that outlive a single
    test. A test that overrides MODEL_CHECKPOINT_PATH must clear both
    before (so the override is actually read) and after (so a later,
    unrelated test doesn't inherit this test's loaded checkpoint)."""
    get_settings.cache_clear()
    inference_provider.reset_inference_service_cache()
    yield
    get_settings.cache_clear()
    inference_provider.reset_inference_service_cache()


@pytest.fixture
def reset_fingerspelling_caches():
    """Same reasoning as reset_inference_caches, for the separate
    fingerspelling classifier's process-wide cache."""
    get_settings.cache_clear()
    fingerspelling_provider.reset_fingerspelling_service_cache()
    yield
    get_settings.cache_clear()
    fingerspelling_provider.reset_fingerspelling_service_cache()
