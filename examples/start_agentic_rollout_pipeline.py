import argparse
import os

from dacite import from_dict
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.agentic_config import AgenticConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    args, extra_overrides = parser.parse_known_args()

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
        initialize(config_path=args.config_path, job_name="app")
        cfg = compose(config_name=args.config_name, overrides=extra_overrides)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()
    from roll.pipeline.agentic.agentic_rollout_pipeline import AgenticRolloutPipeline

    pipeline = AgenticRolloutPipeline(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    main()
