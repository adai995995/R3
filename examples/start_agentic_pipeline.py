import argparse
import os

from dacite import from_dict
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.utils.import_utils import safe_import_class
from roll.utils.str_utils import print_pipeline_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    args, extra_overrides = parser.parse_known_args()

    # Hydra's initialize(config_path=...) resolves relative to the caller module path.
    # This script lives in `R3/examples/`, so passing `examples/foo` would become
    # `examples/examples/foo`. To make CLI usage intuitive, accept:
    # - absolute paths
    # - paths relative to current working directory
    # - paths relative to this script directory (legacy)
    config_path = args.config_path
    if not os.path.isabs(config_path):
        config_path = config_path.lstrip("./")
        if config_path.startswith("examples/"):
            config_path = config_path.removeprefix("examples/")

    cwd_candidate = os.path.abspath(os.path.join(os.getcwd(), config_path))
    script_dir_candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), config_path))

    if os.path.isdir(cwd_candidate):
        initialize_config_dir(config_dir=cwd_candidate, job_name="app")
        cfg = compose(config_name=args.config_name, overrides=extra_overrides)
    elif os.path.isdir(script_dir_candidate):
        initialize_config_dir(config_dir=script_dir_candidate, job_name="app")
        cfg = compose(config_name=args.config_name, overrides=extra_overrides)
    else:
        # Fall back to Hydra's default behavior (may raise a helpful MissingConfigException)
        initialize(config_path=args.config_path, job_name="app")
        cfg = compose(config_name=args.config_name, overrides=extra_overrides)

    ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()

    print_pipeline_config(ppo_config)

    pipeline_cls = getattr(cfg, "pipeline_cls", "roll.pipeline.agentic.agentic_pipeline.AgenticPipeline")
    if isinstance(pipeline_cls, str):
        pipeline_cls = safe_import_class(pipeline_cls)

    pipeline = pipeline_cls(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    main()
