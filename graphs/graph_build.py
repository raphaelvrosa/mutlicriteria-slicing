import os
import json
from graphs.graph_rand import HostCapability, EPACapability, FlowCapability, Topo
from numpy.random import choice
import math
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import itertools


class Graphs:
    FOLDER = './topos/'
    FOLDER_BASE = './topos/base/'

    def __init__(self, range_):
        self.topo = Topo()
        self.range = range_
        self.cap_host = HostCapability(1)
        self.cap_epa = EPACapability(1)
        self.cap_flow = FlowCapability(1)
        self.infra_node_ids = 1
        self.analysis_graph = AnalysisGraph(None)

    def set_profiles(self, profile):
        self.cap_host.set_profile(profile)
        self.cap_epa.set_profile(profile)
        self.cap_flow.set_profile(profile)

    def label_network_nodes(self, graph, label='network', perc_new_label=0.6):
        num_new_labels = int(math.floor(len(graph.nodes())*perc_new_label))
        ids_new_labels = choice(graph.nodes(), size=num_new_labels, replace=False)
        for node_id in ids_new_labels:
            graph.node[node_id]['label'] = label
            graph.node[node_id]['ports'] = 1
            self.fill_node(graph.node[node_id])

    def fill_node(self, data):
        resources = {}
        if data['label'] == 'infra':
            resources = {
                'capabilities': {
                    'host': self.cap_host.get(range=self.range),
                    'epa': self.cap_epa.get(),
                }
            }
        elif data['label'] == 'network':
            resources = {
                'capabilities': {
                    'resources': self.cap_flow.get(range=self.range),
                }
            }
        else:
            pass
        data.update(resources)

    def process_network_rels(self, graph):
        for srcid, dstid, data in graph.edges_iter(data=True):
            src_label = graph.node[srcid]['label']
            dst_label = graph.node[dstid]['label']

            if src_label == 'network' and dst_label == 'network':
                resources = {
                    'label': 'link',
                    'properties': self.cap_flow.get(range=self.range),
                }
                data.update(resources)

    def add_port(self, graph, node_id, port_data, link_data):
        src_ports = graph.node[node_id]['ports']
        node_label = graph.node[node_id]['label']
        src_port_id = node_label + '-' + str(node_id) + '-port-' + str(src_ports)
        graph.node[node_id]['ports'] = src_ports + 1
        graph.add_node(src_port_id, **port_data)
        link_data = link_data if link_data else {'label': 'Port'}
        graph.add_edge(node_id, src_port_id, **link_data)
        return src_port_id

    def add_node_ports(self, graph, node_id, num_ports, port_type, node_type='infra'):
        port_ids = []
        for port_id in range(num_ports):
            if node_type == 'infra':
                link_data = {'label': 'port'}
            else:
                link_data = {
                    'label': 'link',
                    'properties': self.cap_flow.get(range=self.range),
                }
            data = {
                'label': 'port',
                'properties': {
                    'port_id': port_id,
                    'port_type': port_type,
                }
            }
            add_port_id = self.add_port(graph, node_id, data, link_data)
            port_ids.append(add_port_id)
        return port_ids

    def add_graph_ports(self, graph, number_ports=1):
        shortest_paths = nx.floyd_warshall(graph)
        pair_shortest_paths = {}
        for n in shortest_paths:
            max_n_dist = max(shortest_paths[n].values())
            ind_max_n_dist = shortest_paths[n].values().index(max_n_dist)
            d = shortest_paths[n].keys()[ind_max_n_dist]
            pair_shortest_paths[(n,d)] = max_n_dist

        sorted_paths = sorted(pair_shortest_paths, key=pair_shortest_paths.get, reverse=True)
        (src, dst) = sorted_paths[0]
        # print 'ennis src, dst', src, dst
        for node_id in (src, dst):
            port_ids = self.add_node_ports(graph, node_id, number_ports, 'enni', node_type='network')
            self.add_network_ports(graph, node_id, port_ids)

    def add_network_ports(self, graph, net_id, port_ids):
        link_data = {
            'label': 'link',
            'properties': self.cap_flow.get(range=self.range),
        }
        for port_id in port_ids:
            graph.add_edge(net_id, port_id, **link_data)

    def add_infra_node(self, graph):
        data = {
            'label': 'infra',
            'capabilities': {
                'host': self.cap_host.get(range=self.range),
                'epa': self.cap_epa.get(),
            }
        }
        node_id = 'infra_' + str(self.infra_node_ids)
        graph.add_node(node_id, data)
        self.infra_node_ids += 1
        return node_id

    def add_edge_net(self, graph, src, dst):
        data = {
            'label':'link',
            'properties': self.cap_flow.get(range=self.range),
        }
        graph.add_edge(src, dst, **data)

    def select_node_ids(self, graph, field='label', label='infra'):
        node_ids = [n for n, d in graph.nodes_iter(data=True) if d[field] == label]
        return node_ids

    def add_graph_infra(self, graph, infra_rate, infra_mode):
        node_ids_net = self.select_node_ids(graph, label='network')
        number_nodes = int(math.ceil(len(node_ids_net) * infra_rate))

        if infra_mode == 'vitality':
            metrics = nx.closeness_vitality(graph)
            sorted_metrics = sorted(metrics, key=metrics.get, reverse=True)
            ids_infra = [x for x in sorted_metrics if graph.node[x]['label'] == 'network']
            ids_infra = ids_infra[:number_nodes]
            # ids_infra = sorted_metrics[:number_nodes]
        elif infra_mode == 'centrality':
            metrics = nx.closeness_centrality(graph)
            sorted_metrics = sorted(metrics, key=metrics.get, reverse=True)
            # ids_infra = sorted_metrics[:number_nodes]
            ids_infra = [x for x in sorted_metrics if graph.node[x]['label'] == 'network']
            ids_infra = ids_infra[:number_nodes]
        else:
            ids_infra = choice(node_ids_net, size=number_nodes, replace=False)

        for node_id in ids_infra:
            infra_id = self.add_infra_node(graph)
            self.add_edge_net(graph, node_id, infra_id)

    def get_graph(self, model, model_kwargs, profile, infra_nodes_rate, infra_mode='vitality'):
        self.set_profiles(profile)
        graph = self.topo.create(model, model_kwargs)
        self.label_network_nodes(graph, perc_new_label=1.0)
        self.process_network_rels(graph)
        self.add_graph_ports(graph)
        self.add_graph_infra(graph, infra_nodes_rate, infra_mode)
        return graph

    def get_graph_from_raw(self, size, model, model_kwargs, profile, infra_nodes_rate, infra_mode='vitality'):
        filename = 'topo_' + str(size) + '_' + str(model) + '_' + str(profile)
        graph = self.retrieve_graph(filename, base=True)
        self.add_graph_infra(graph, infra_nodes_rate, infra_mode)
        return graph

    def get_graph_raw(self, model, model_kwargs, profile):
        self.set_profiles(profile)
        graph = self.topo.create(model, model_kwargs)
        self.label_network_nodes(graph, perc_new_label=1.0)
        self.process_network_rels(graph)
        self.add_graph_ports(graph)
        return graph

    def parse_filename(self, filename, base=False):
        if base:
            filen = self.FOLDER_BASE + filename + '.json'
        else:
            filen = self.FOLDER + filename + '.json'
        return filen

    def readfile(self, filename, base):
        filename = self.parse_filename(filename, base=base)
        with open(filename, 'r') as infile:
            data = json.load(infile)
            return data

    def writefile(self, data, filename):
        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=4, sort_keys=True)
            return True

    def save_graph(self, graph, filename, parse_filename=True, base=False):
        # print 'saving ', filename
        if parse_filename:
            filename = self.parse_filename(filename, base=base)
        data = json_graph.node_link_data(graph)
        if self.writefile(data, filename):
            return True
        return False

    def retrieve_graph(self, filename, base=False):
        data = self.readfile(filename, base=base)
        graph = json_graph.node_link_graph(data)
        return graph

    def to_raw(self, g):
        for src, dst, data in g.edges_iter(data=True):
            if data['label'] == 'link' or data['label'] == 'virtuallink':
                props = data.get('properties', None)
                if props:
                    data.update(props)
                    del data['properties']
        for node,data in g.nodes_iter(data=True):
            if data['label'] == 'infra':
                capabs = data.get('capabilities', None)
                if capabs:
                    props_host = capabs.get('host', None)
                    if props_host:
                        data.update(props_host)
                    
                    props_ana = capabs.get('analysis', None)
                    if props_ana:
                        data.update(props_ana)
    
                    del data['capabilities']
            if data['label'] == 'network':
                capabs = data.get('capabilities', None)
                if capabs:
                    props_ana = capabs.get('analysis', None)
                    if props_ana:
                        data.update(props_ana)
                    del data['capabilities']

    def graph_analysis(self, graph):
        self.analysis_graph.set_graph(graph)
        self.analysis_graph.analysis()
        self.to_raw(graph)
        return graph

    def draw(self, graph):
        nx.draw(graph, with_labels=True)
        plt.show()


    def pairwise_nodes(self, candidate_sets):
        '''

        :param candidate_sets: list of lists
        :return: list of combinations
        '''
        combination = list(itertools.product(*candidate_sets))
        return combination

    def get_ennis(self, graph):
        ennis = {}
        for src, data in graph.nodes_iter(data=True):
            label_src = data['label']
            if label_src == 'network':
                for dst in graph.neighbors(src):
                    label_dst = graph.node[dst]['label']
                    if label_dst == 'port':
                        if src not in ennis:
                            ennis[src] = []
                        ennis[src].append(dst)
        return ennis

    def select_ids(self, g, node_feats):
        '''
        :param g: graph
        :param cap_type: epa or host
        :param req_caps: list of caps must exist
        :return: node ids with req caps
        '''
        selection = []
        for node, data in g.nodes_iter(data=True):
            if data['label'] == 'infra':
                good_res = True
                good_caps = True

                req_hosts = node_feats
                for req_host, req_res in req_hosts.items():
                    req_cap = req_res.get('host')
                    exist_cap = data
                    
                    if not self.check_caps('host', exist_cap, req_cap):
                        good_res = False
                    else:
                        good_res = True
                        break

                if good_res and good_caps:
                    selection.append(node)
        return selection

    def get_network_id_adj(self, graph, nodes):
        adj_nets = {}
        for node_id in nodes:
            for neigh in graph.neighbors(node_id):
                if graph.node[neigh]['label'] == 'network':
                    adj_nets[neigh] = graph.node[neigh]
        return adj_nets

    def check_caps(self, cap_type, exist_caps, req_caps):
        if cap_type == 'epa':
            return all([exist_caps[req] for req in req_caps])
        if cap_type == 'host':
            return all([exist_caps[req] >= req_caps[req] for req in req_caps])
        return False


class AnalysisGraph:
    def __init__(self, graph):
        self.set_graph(graph)

    def set_graph(self, graph):
        self.graph = graph

    def analysis(self):
        avnd = nx.average_neighbor_degree(self.graph)
        vit = nx.closeness_vitality(self.graph)
        cent = nx.closeness_centrality(self.graph)
        bet = nx.betweenness_centrality(self.graph)
        clus = nx.clustering(self.graph)
        # ecc = nx.eccentricity(self.graph)

        for node,data in self.graph.nodes_iter(data=True):
            label = data['label']
            if label == 'infra' or label == 'network':
                cap = {
                    # 'eccentricity': ecc[node],
                    'betweenness': bet[node],
                    'centrality': cent[node],
                    'vitality': vit[node],
                    'avg_neigh_degree': avnd[node],
                    'clustering': clus[node]
                }
                if  'capabilities' in data:
                    data['capabilities']['analysis'] = cap

        self.cohesion_index()

    def cohesion_index(self):
        def cohesion(src, dst):
            n_src = self.graph.neighbors(src)
            n_dst = self.graph.neighbors(dst)

            all_neighbors = float(len(n_src) + len(n_dst))
            common_neighbors = sum( [1 if n in n_src else 0 for n in n_dst ] )
            if all_neighbors == 0:
                return 0
            else:
                return common_neighbors/all_neighbors

        for src in self.graph.nodes_iter():
            for dst in self.graph.neighbors(src):
                if 'cohesion' not in self.graph.edge[src][dst]:
                    src_dst = cohesion(src, dst)
                    self.graph.edge[src][dst]['cohesion'] = src_dst




if __name__ == "__main__":
    model = 4
    model_kwargs = {'nodes': 10, 'degree': 5, 'neighbour_edges': 5, 'edge_prob': 0.5}
    profile = 3
    infra_nodes_rate = 0.2
    infra_mode = 1

    graphs = Graphs(range_=True)
    graph = graphs.get_graph(model, model_kwargs, profile, infra_nodes_rate, infra_mode)
    graph = graphs.graph_analysis(graph)

    # print '$$$$$$$$ NODES $$$$$$$$$'
    # for nid, data in graph.nodes_iter(data=True):
    #     print nid,data
    
    # print '$$$$$$$$ EDGES $$$$$$$$$'
    # for srcid, dstid, data in graph.edges_iter(data=True):
    #     print srcid, dstid, data

    # graphs.draw(graph)

