import json
from dataclasses import asdict

from dreamer.config import Config
from dreamer.inspect import infer_config_from_checkpoint


def test_inspector_ignores_retired_actor_warmup_in_historical_config(tmp_path):
    run_dir = tmp_path / "historical_run"
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_final.pt"
    checkpoint_path.parent.mkdir(parents=True)
    data = asdict(Config())
    data["actor_warmup_steps"] = 3000
    (run_dir / "config.json").write_text(json.dumps(data))

    config = infer_config_from_checkpoint(checkpoint_path, config_name=None)

    assert not hasattr(config, "actor_warmup_steps")
    assert config.d_hidden == Config().d_hidden
