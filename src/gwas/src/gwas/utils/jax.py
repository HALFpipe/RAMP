import jax


def setup_jax() -> None:
    jax.config.update("jax_enable_x64", True)

    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
