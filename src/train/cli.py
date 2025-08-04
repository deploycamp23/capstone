from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from train.config import TrainingConfig, apply_overrides, load_config, save_config
from train.train_core import save_artifacts, train


def parse_keyvals(pairs: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise click.BadParameter("Override must be in key=value format")
        k, v = item.split("=", 1)
        out[k] = v
    return out


@click.command()
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--set", "overrides", multiple=True, help="Override config values as key=value", metavar="key=value")
@click.option("--save", is_flag=True, help="Persist effective config back to file")
def main(config_path: Path, overrides: tuple[str, ...], save: bool) -> None:
    cfg = load_config(config_path)
    ov = parse_keyvals(overrides)
    cfg = apply_overrides(cfg, ov)

    result = train(cfg)
    save_artifacts(cfg.output_dir, result["model"], result["metrics"], result["history"])

    if save:
        save_config(cfg, config_path)


if __name__ == "__main__":
    main()
