import pytest
import yaml

from interptemp.config import load_config


@pytest.fixture
def cfg_path(tmp_path):
    (tmp_path / "m.yaml").write_text(yaml.safe_dump({"name": "org/model", "dtype": "float32"}))
    (tmp_path / "m2.yaml").write_text(yaml.safe_dump({"name": "org/other"}))
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump({"name": "x", "target": "a:B", "model": "m.yaml", "params": {"k": 1}})
    )
    return p


def test_model_path_relative_to_config(cfg_path):
    cfg = load_config(cfg_path)
    assert cfg.model.name == "org/model"
    assert cfg.model.dtype == "float32"


def test_overrides_parse_yaml_values(cfg_path):
    cfg = load_config(
        cfg_path,
        ["params.k=3", "params.new.deep=[1, 2]", "generation.do_sample=true", "judge=null"],
    )
    assert cfg.params == {"k": 3, "new": {"deep": [1, 2]}}
    assert cfg.generation.do_sample is True
    assert cfg.judge is None


def test_model_swap_then_field_override(cfg_path):
    cfg = load_config(cfg_path, ["model.dtype=bfloat16", f"model={cfg_path.parent / 'm2.yaml'}"])
    assert cfg.model.name == "org/other"
    assert cfg.model.dtype == "bfloat16"


def test_unknown_field_rejected(cfg_path):
    with pytest.raises(ValueError):
        load_config(cfg_path, ["generation.max_tokens=3"])  # typo for max_new_tokens


def test_bad_override_format(cfg_path):
    with pytest.raises(ValueError):
        load_config(cfg_path, ["params.k"])
