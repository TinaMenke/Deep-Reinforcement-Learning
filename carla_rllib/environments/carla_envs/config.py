

class BaseConfig(object):

    def __init__(self, config=None):
        if config:
            self.parse(config)

    def parse(self, config):
        self.agent_type = config["agent_type"]
        self.frame_skip = config["frame_skip"]
        #self.frame_skip = 4
        #self.frame_skip = 2
        self.stable_baselines = config["stable_baselines"]
        self.sync_mode = config["sync_mode"]
        self.delta_sec = config["delta_sec"]
        self.render = config["render"]
        self.host = config["host"]
        self.port = config["port"]
        self.map = config["map"]
        self.num_agents = config["num_agents"]

    def __repr__(self):
        return ("Agent type: %s\n" +
                "Number of Agents: %d\n" +
                "Sync Mode: %s\n" +
                "Rendering: %s\n" +
                "Host: %s\n" +
                "Port: %d\n" +
                "Map: %s\n" +
                "Frame skipping (Continuous Agent only): %s\n" +
                "Seconds between Frames (Sync mode only): %s\n" +
                "Baselines support (Single agent only): %s") % (self.agent_type,
                                                                self.num_agents,
                                                                self.sync_mode,
                                                                self.render,
                                                                self.host,
                                                                self.port,
                                                                self.map,
                                                                self.frame_skip,
                                                                self.delta_sec,
                                                                self.stable_baselines,
                                                                )


def parse_json(json_dict):
    core = zip(json_dict["hosts"], json_dict["ports"],
               json_dict["maps"], json_dict["num_agents"])
    configs = []
    for host, port, map, num_agents in core:
        config_dict = dict(
            agent_type=json_dict["agent_type"],
            frame_skip=json_dict["frame_skip"],
            stable_baselines=json_dict["stable_baselines"],
            sync_mode=json_dict["sync_mode"],
            delta_sec=json_dict["delta_sec"],
            render=json_dict["render"],
            host=host,
            port=port,
            map=map,
            num_agents=num_agents)
        config = BaseConfig(config_dict)
        configs.append(config)
    return configs
