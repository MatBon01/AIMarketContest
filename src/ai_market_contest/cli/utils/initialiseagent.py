import configparser
import datetime
import pathlib
from string import Template
import typing

from ai_market_contest.cli.cli_config import (  # type: ignore
    AGENT_FILE,
    AGENT_TEMPLATE,
    ABS_METHOD_STR,
    IMPORT_STR,
    AGENT_STR,
    CLASS_METHOD_STR,
    TRAINED_AGENTS_DIR_NAME,
    INITIAL_PICKLER_FILE,
    INITIAL_PICKLER_NAME,
    CONFIG_FILENAME
)
from ai_market_contest.cli.utils.processmetafile import write_meta_file
from ai_market_contest.cli.utils.filesystemutils import check_config_file_exists


def set_agent_to_initialised(agent_dir: pathlib.Path): 
    config_file: pathlib.Path = agent_dir / CONFIG_FILENAME
    check_config_file_exists(config_file)    
    config: configparser.ConfigParser = configparser.ConfigParser()
    config.read(config_file)
    config["training"]["initialised"] = "True"
    with config_file.open("w") as c_file:
        config.write(c_file)
    
    

def make_initial_trained_agent(agent_dir: pathlib.Path, agent_name: str, initial_hash: str):
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


def make_agent_classname_camelcase(agent_name: str):
    AGENT_STR = "agent"
    if AGENT_STR.capitalize() in agent_name:
        return agent_name
    agent_name_cc = agent_name.lower()
    if AGENT_STR in agent_name_cc:
        agent_name_cc = agent_name_cc.replace(AGENT_STR, AGENT_STR.capitalize())
    return agent_name_cc[0].upper() + agent_name_cc[1:]


def create_new_agent_file(agent_file: pathlib.Path, agent_name: str):
    subs: typing.Dict[str, str] = {"agent_classname": make_agent_classname_camelcase(agent_name)}
    with AGENT_FILE.open("r") as a_file:
        src = Template(a_file.read())
    with agent_file.open("w") as new_agent_file:
        new_agent_file.write(src.substitute(subs))
