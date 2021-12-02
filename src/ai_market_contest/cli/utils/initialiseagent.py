import configparser
import datetime
import pathlib
import sys
import typing
from string import Template

from ai_market_contest.cli.cli_config import (  # type: ignore
    AGENT_FILE,
    AGENTS_DIR_NAME,
    CONFIG_FILENAME,
    INITIAL_PICKLER_FILE,
    INITIAL_PICKLER_NAME,
    TRAINED_AGENTS_DIR_NAME,
)
from ai_market_contest.cli.utils.filesystemutils import check_config_file_exists
from ai_market_contest.cli.utils.hashing import set_agent_initial_hash
from ai_market_contest.cli.utils.processmetafile import write_meta_file


def set_agent_to_initialised(agent_dir: pathlib.Path):
    config_file: pathlib.Path = agent_dir / CONFIG_FILENAME
    check_config_file_exists(config_file)
    config: configparser.ConfigParser = configparser.ConfigParser()
    config.read(config_file)
    config["training"]["initialised"] = "True"
    with config_file.open("w") as c_file:
        config.write(c_file)


def make_initial_trained_agent(
    agent_dir: pathlib.Path, agent_name: str, initial_hash: str
):
    trained_agents_dir = agent_dir / TRAINED_AGENTS_DIR_NAME
    initial_trained_agent_dir = trained_agents_dir / initial_hash
    initial_trained_agent_dir.mkdir(parents=True)
    msg = "Initial untrained agent"
    write_meta_file(
        initial_trained_agent_dir, initial_hash, datetime.datetime.now(), msg
    )
    new_pickler_file: pathlib.Path = agent_dir / INITIAL_PICKLER_NAME
    subs: typing.Dict[str, str] = {
        "agent_import": agent_name,
        "agent_classname": make_agent_classname_camelcase(agent_name),
    }
    with INITIAL_PICKLER_FILE.open("r") as initial_pickler:
        src = Template(initial_pickler.read())

    with new_pickler_file.open("w") as new_pickler:
        new_pickler.write(src.substitute(subs))


def create_agent_class(
    agent_name: str, proj_dir: pathlib.Path, check_overwrite: bool = False
):
    agents_dir = proj_dir / AGENTS_DIR_NAME
    agent_filename: str = agent_name + ".py"
    agent_dir: pathlib.Path = agents_dir / agent_name
    if check_overwrite:
        check_overwrite_agent(agent_filename, agent_dir)
    agent_file: pathlib.Path = agent_dir / agent_filename
    agent_dir.mkdir(parents=True)
    agent_file.touch()
    create_new_agent_file(agent_file, agent_name)
    initial_hash: str = set_agent_initial_hash(agent_dir)
    make_initial_trained_agent(agent_dir, agent_name, initial_hash)


def check_overwrite_agent(agent_filename: str, agent_dir: pathlib.Path):
    if agent_dir.is_dir():
        overwrite = "x"
        while overwrite != "y" and overwrite != "n":
            overwrite = input(
                f"{agent_filename} already exists, are you sure you want"
                + " to override the existing file? (y/n): "
            )
            if overwrite == "y":
                break
            if overwrite == "n":
                sys.exit(0)


def make_agent_classname_camelcase(agent_name: str):
    AGENT_STR = "agent"
    if AGENT_STR.capitalize() in agent_name:
        return agent_name
    agent_name_cc = agent_name.lower()
    if AGENT_STR in agent_name_cc:
        agent_name_cc = agent_name_cc.replace(AGENT_STR, AGENT_STR.capitalize())
    return agent_name_cc[0].upper() + agent_name_cc[1:]


def create_new_agent_file(agent_file: pathlib.Path, agent_name: str):
    subs: typing.Dict[str, str] = {
        "agent_classname": make_agent_classname_camelcase(agent_name)
    }
    with AGENT_FILE.open("r") as a_file:
        src = Template(a_file.read())
    with agent_file.open("w") as new_agent_file:
        new_agent_file.write(src.substitute(subs))