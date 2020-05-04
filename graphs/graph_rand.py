from numpy import random
import networkx as nx
from .defs import *

profiles = {
    1: INFRA_PROFILE_1,
    2: INFRA_PROFILE_2,
    3: INFRA_PROFILE_3,
    4: INFRA_PROFILE_4,
}

nf_profiles = {
    1: NF_PROFILE_1,
    2: NF_PROFILE_2,
    3: NF_PROFILE_3,
    4: NF_PROFILE_4,
}

class Randomness:
    def __init__(self, _min, _max):
        self.min = _min
        self.max = _max
        self.value = 0

    def set_range(self, _min, _max):
        self.min = _min
        self.max = _max

    def get(self, _min=None, _max=None, range=False, values=None):
        if range and values:
            self.value = random.choice(values, replace=False)
        elif _min and _max:
            self.set_range(_min, _max)
            self.value = random.randint(self.min, self.max)
        return self.value

    def get_bool(self):
        self.value = random.choice([True, False])
        return self.value


class HostCapability:
    def __init__(self, profile, profile_nf=False):
        self.r = Randomness(0,0)
        self.values = {
            'num_cpus':0,
            'mem_size':0,
            'disk_size':0,
        }
        if profile_nf:
            self.resources_range = nf_profiles[profile]
            # self.set_profile(profile)
        else:
            self.resources_range = profiles[profile]
            # self.set_profile(profile)

    def set_profile(self, profile):
        self.resources_range = profiles[profile]
        self.cpu_min, self.cpu_max = self.resources_range['NODE_CPU']
        self.mem_min, self.mem_max = self.resources_range['NODE_MEM']
        self.storage_min, self.storage_max = self.resources_range['NODE_STORAGE']

    def get(self, range=False, suffix=False):
        if range:
            cpus = self.r.get(range=range, values=self.resources_range['NODE_CPU_RANGE'])
            mem = self.r.get(range=range, values=self.resources_range['NODE_MEM_RANGE'])
            disk = self.r.get(range=range, values=self.resources_range['NODE_STORAGE_RANGE'])

            self.values = {
                'num_cpus': str(cpus) if suffix else cpus,
                'mem_size': str(mem) + ' GB' if suffix else mem,
                'disk_size': str(disk) + ' GB' if suffix else disk,
            }
        else:
            cpus = self.r.get(self.cpu_min, self.cpu_max)
            mem = self.r.get(self.mem_min, self.mem_max)
            disk = self.r.get(self.storage_min, self.storage_max)
            self.values = {
                'num_cpus': str(cpus) if suffix else cpus,
                'mem_size': str(mem) + ' GB' if suffix else mem,
                'disk_size': str(disk) + ' GB' if suffix else disk,
            }
        return self.values


class EPACapability:
    def __init__(self, profile):
        # http://events.linuxfoundation.org/sites/events/files/slides/NFV%20Orchestration%20for%20Optimal%20Performance.pdf
        self.r = Randomness(0,0)
        self.profile = profile
        self.values = {
            "ovs_acceleration": bool(self.r.get_bool()),
            "huge_page": bool(self.r.get_bool()),
            "cpu_pinning": bool(self.r.get_bool()),
            "SR_IOV": bool(self.r.get_bool()),
            "PCI": bool(self.r.get_bool()),
            "NUMA": bool(self.r.get_bool()),
            "DPDK": bool(self.r.get_bool()),
            "AES": bool(self.r.get_bool()),  # andvanced encryption
            "CAT": bool(self.r.get_bool()),  # cache allocation
            "TXT": bool(self.r.get_bool()),  # trusted compute pools
        }

    def set_profile(self, profile):
        self.profile = profile

    def get(self, profile=10, alloc=False):
        self.values = {
            "ovs_acceleration": bool(self.r.get_bool()),
            "huge_page": bool(self.r.get_bool()),
            "cpu_pinning": bool(self.r.get_bool()),
            "SR_IOV": bool(self.r.get_bool()),
            "PCI": bool(self.r.get_bool()),
            "NUMA": bool(self.r.get_bool()),
            "DPDK": bool(self.r.get_bool()),
            "AES": bool(self.r.get_bool()),  # andvanced encryption
            "CAT": bool(self.r.get_bool()),  # cache allocation
            "TXT": bool(self.r.get_bool()),  # trusted compute pools
        }

        if profile is not None:
            self.profile = profile

        values = {}
        keys = self.values.keys()

        if alloc:
            rand_keys = random.choice(keys, self.profile, replace=False)
            for key in rand_keys:
                values[key] = True
        else:
            for i in range(self.profile):
                key = keys[i]
                # values[key] = self.values[key]
                values[key] = True
        return values


class FlowCapability:
    def __init__(self, profile):
        self.r = Randomness(0, 0)
        self.values = {
            "latency": 0,
            "availability": 1,
            "frame_loss_ratio": 0.0,
            "throughput": 0,
        }
        self.resources_range = profiles[profile]

    def set_profile(self, profile):
        self.resources_range = profiles[profile]

    def get(self, range=False):
        if range:
            self.values = {
                "latency": self.r.get(range=range, values=self.resources_range['LINK_DELAY_RANGE']),
                "availability": float(self.r.get(*self.resources_range['LINK_AVAILABILITY']) / 10000.0),
                "frame_loss_ratio": float(self.r.get(*self.resources_range['LINK_FRAME_LOSS']) / 1000.0),
                "throughput": self.r.get(range=range, values=self.resources_range['LINK_BANDWIDTH_RANGE'])*1000,
            }
        else:
            self.values = {
                "latency": self.r.get(*self.resources_range['LINK_DELAY']),
                "availability": float(self.r.get(*self.resources_range['LINK_AVAILABILITY'])/10000.0),
                "frame_loss_ratio": float(self.r.get(*self.resources_range['LINK_FRAME_LOSS'])/1000.0),
                "throughput": self.r.get(*self.resources_range['LINK_BANDWIDTH'])*1000,
            }
        return self.values


class Topo:
    def __init__(self):
        self.nodes = NODES
        self.degree = DEGREE
        self.edge_prob = EDGES_PROB
        self.neighbour_edges = NEIGHBOUR_EDGES
        self.graph = None

    def create(self, model, kwargs):
        if model == 1:
            degree = kwargs.get("degree", self.degree)
            nodes = kwargs.get("nodes", self.nodes)
            self.graph = nx.random_regular_graph(degree, nodes)
        elif model == 2:
            nodes = kwargs.get("nodes", self.nodes)
            edge_prob = kwargs.get("edge_prob", self.edge_prob)
            self.graph = nx.binomial_graph(nodes, edge_prob)
        # elif model == 3:
        #     self.graph = nx.gaussian_random_partition_graph(self.nodes, self.nodes/self.neighbour_edges,
        #                                                     self.nodes / self.neighbour_edges,
        #                                                     self.edge_prob, self.edge_prob)
        elif model == 3:
            nodes = kwargs.get("nodes", self.nodes)
            neighbour_edges = kwargs.get("neighbour_edges", self.neighbour_edges)
            edge_prob = kwargs.get("edge_prob", self.edge_prob)
            self.graph = nx.powerlaw_cluster_graph(nodes, neighbour_edges, edge_prob)
        elif model == 4:
        #     self.graph = nx.scale_free_graph(self.nodes)
        # else:
            nodes = kwargs.get("nodes", self.nodes)
            neighbour_edges = kwargs.get("neighbour_edges", self.neighbour_edges)
            self.graph = nx.barabasi_albert_graph(nodes, neighbour_edges)

        return self.graph


if __name__ == '__main__':
    topos = Topo()
    g = topos.get(5)
    # print nx.is_directed(g)
    # print nx.is_connected(g)
    # print g.is_multigraph()
