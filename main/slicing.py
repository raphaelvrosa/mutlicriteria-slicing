import json
import os
import itertools

import pandas as pd
import seaborn as sns
import numpy as np

import networkx as netx
import matplotlib.pyplot as plt

from graphs.graph_build import Graphs, AnalysisGraph
from graphs.graph_draw import GraphDraw, GraphView
from networkx.readwrite import json_graph

from mca.topsys import TOPSIS
from mca.ahp import AHP

from functools import reduce

import gevent


class Topologies:
    def __init__(self):
        self.folder = './topos/'
        self.graphs = Graphs(range_=True)
        
    def create_topo_base(self, model, model_kwargs, infra_profile):
        self.graphs = Graphs(range_=True)
        graph = self.graphs.get_graph_raw(model, model_kwargs, infra_profile)
        graph = self.graphs.graph_analysis(graph)
        return graph

    def create_topo(self, model, model_kwargs, profile, infra_nodes_rate, infra_mode):
        self.graphs = Graphs(range_=True)
        graph = self.graphs.get_graph(model, model_kwargs, profile, infra_nodes_rate, infra_mode)
        graph = self.graphs.graph_analysis(graph)
        return graph

    def create_topo_from_raw(self, model, size, model_kwargs, profile, infra_nodes_rate, infra_mode):
        self.graphs = Graphs(range_=True)
        graph = self.graphs.get_graph_from_raw(size, model, model_kwargs, profile, infra_nodes_rate, infra_mode)
        self.graphs.to_raw(graph)
        return graph

    def store(self, graph, filename, base=False):
        self.graphs.save_graph(graph, filename, base=base)
        
    def load(self, filename, base=False):
        graph = self.graphs.retrieve_graph(filename, base=base)
        return graph

    def candidate_paths(self, g, nodes_chain):
        srcs_dsts = self.map_srcs_dsts(g)
        chain_caps = self.candidate_cap_nodes(g, nodes_chain)
        if chain_caps:
            # print 'chain infra node ids', chain_caps
            # print 'port id srcs_dsts', srcs_dsts
            path_seq_candidates = self.mix_ennis_chains(srcs_dsts, chain_caps)
            multiple_choice_paths = self.graphs.pairwise_nodes(path_seq_candidates)
            return multiple_choice_paths
        # print 'no candidate chains of nodes found for requested capabilities'
        return []

    def mix_ennis_chains(self, srcs_dsts, chain_caps):
        srcs = list(set(srcs_dsts.keys()))
        dsts = list(set(srcs_dsts.values()))
        path_seq_candidates = []
        path_seq_candidates.append(srcs)
        path_seq_candidates.extend(chain_caps)
        path_seq_candidates.append(dsts)
        return path_seq_candidates

    def map_srcs_dsts(self, g, connectivity='a2a'):
        srcs_dsts = {}
        ennis = self.graphs.get_ennis(g)
        src_nets = []
        # print 'ennis', ennis
        if connectivity == 'a2a':
            for netsrc in ennis.keys():
                src_nets.append(netsrc)
                for netdst in ennis.keys():
                    if netdst not in src_nets:
                        srcs_dsts[netsrc] = netdst

        elif connectivity == 'o2a':
            for netsrc in ennis.items():
                src_nets.append(netsrc)
                for netdst in ennis.items():
                    if netdst not in src_nets:
                        srcs_dsts[netsrc] = netdst
                break
        return srcs_dsts

    def candidate_cap_nodes(self, g, cand_nodes):
        chain_caps = []
        # print cand_nodes
        for node in cand_nodes:
            cand_infra_nodes = self.graphs.select_ids(g, node)
            if not cand_infra_nodes:
                return []
            chain_caps.append(cand_infra_nodes)
        return chain_caps


class Slices:
    def __init__(self):
        self.slices = {}
        self.metrics = {}
        self.ennis = {}
        self.weight_pattern = None
        self.link_metrics = {}

    def extract(self, prop, func, data_props):
        extracted_props_list = [data[prop] for data in data_props]
        value = func(extracted_props_list)
        return value

    def process_path_data(self, data_props):
        link_metrics = {
            'throughput': min,
            'latency': max,
            'availability': min,
            'frame_loss_ratio': max,
        }
        props = {}
        for prop, func in link_metrics.items():
            props[prop] = self.extract(prop, func, data_props)
        return props

    def shortcut_func(self, G, path):
        data_props = []
        for srcid in range(len(path[:-1])):
            src = path[srcid]
            dst = path[srcid + 1]
            data = G.edge[src][dst]
            data_props.append(data)
        new_data = self.process_path_data(data_props)
        return new_data

    def preprocess_graph(self, graph, weight_metric, max_hops=3):
        levels, ranks, arcs = preprocess(graph, weight_metric, self.shortcut_func, max_hops)
        return levels, ranks, arcs

    def slice(self, graph, levels, candidate_chains, metric):
        # results = self.pairwise_dists(graph, levels, candidate_chains, metric)
        # paths = self.process_results(results)
        paths = self.pairwise_dists(graph, levels, candidate_chains, metric)
        return paths

    def pairwise_dists(self, graph, levels, candidate_chains, metric):
        results = {}
        for candidate_chain in candidate_chains:
            jobs = self.create_dist_jobs(graph, levels, candidate_chain, metric)
            gevent.joinall(jobs)
            result = [job.value for job in jobs]
            # print 'candidate_chain', candidate_chain, 'result', result
            results[candidate_chain] = result
        return results

    def merge_paths(self, seqs):
        a = []
        for seq in seqs:
            a.extend(seq[:-1])
        a.append(seqs[-1][-1])
        return a

    def process_results(self, results):
        paths = {}
        for chain, result in results.items():
            # dists = [item[0] for item in result]
            # seqs = [item[1] for item in result]
            dists = [len(item)-1.0 for item in result]
            seqs = result
            paths[chain] = (dists, seqs)
        return paths

    def create_dist_jobs(self, graph, levels, candidate_chain, metric):
        jobs = []
        for src_id in range(len(candidate_chain[1:])):
            src = candidate_chain[src_id]
            dst = candidate_chain[src_id+1]
            # job = gevent.spawn(contract_hierarchy, graph, src, dst, levels, metric)
            job = gevent.spawn(netx.shortest_path, graph, source=src, target=dst)
            jobs.append(job)
        return jobs


class Analysis:
    def __init__(self):
        self.topos = Graphs(True)
        self.slicing = Slices()
        self.g = None
        self.metrics = []
        self.srcs_dsts = {}
        self.paths = {}
        self.shortcuts = {}
        self.levels = {}

    def get_graph(self):
        return self.g

    def set_graph(self, g):
        self.g = g

    # def structure(self, model):
    #     self.g = self.topos.graph_network_ennis(model, range=True)
    #     self.topos.set_analysis_metrics(self.g)

        '''

        :param service_chain (list): src enni, list of NFs containing EPAs, dst enni
        :return: combination of possible infra sequence nodes
        '''

        # self.slicing.set_link_metrics(link_patterh)
        # self.slicing.set_weight_pattern(metric)

    def connect(self, g, candidate_chains, metric):
        # if not self.levels:
        self.levels = self.init_levels(g)
        self.paths = self.slicing.slice(g, self.levels, candidate_chains, metric)
        return self.paths

    def init_levels(self, g):
        level = {}
        for n in g.nodes_iter():
            level[n] = 0
        return level

    def augment(self, g, metric):
        self.levels, ranks, self.shortcuts = self.slicing.preprocess_graph(g, metric, max_hops=3)

    def quality_path(self, path, metric):
        cost = path_cost(self.g, path, weight=metric)
        return cost

    def quality_shortcuts(self, path):
        full_path,used_shorcuts = restore_full_path(path, self.shortcuts)
        return full_path,used_shorcuts

    def qualify(self, paths, metric):
        path = self.slicing.merge_paths(paths)
        path_cost = self.quality_path(path, metric)
        full_path,path_shorts = self.quality_shortcuts(path)
        # print path_cost, path_shorts, full_path

    def qualify_path_mappings(self, graph, paths):
        paths_maps = {}
        for candidate in paths.keys():
            cand = candidate[1:-1]
            adj_nets = self.topos.get_network_id_adj(graph, cand)
            paths_maps[cand] = adj_nets
        return paths_maps

    def qualify_paths(self, paths):
        paths_feats = {}
        for candidate, path_seqs in paths.items():
            # dists, path_seqs = value
            edge_feats = {}
            for seq in path_seqs:
            # path = self.slicing.merge_paths(path_seqs)
                if len(seq) == 1:
                    seq.append(seq[0])
                # full_path, _ = restore_full_path(seq, self.shortcuts)
                meas = self.measures(seq)
                edge_feats[tuple(seq)] = meas

            path_features = self.measures_paths(candidate, edge_feats)
            paths_feats[candidate] = path_features
        return paths_feats

    def measures_paths(self, cand_path, edges_feats):
        '''
        Create full features vector of nodes and edges inside path
        :param candidate:
        :param path_feats:
        :return:
        '''
        node_feats = {}
        for cand in cand_path:
            nf = self.measures_node(cand)
            node_feats[cand] = nf
        # alt = list(zip(node_feats,edges_feats))
        # alt = [list(el) for el in alt]
        # path_feats = sum(alt, [])
        # path_feats = sum(path_feats, [])
        # path_feats.extend(node_feats[-1])

        path_feats = {
            'nodes': node_feats,
            'edges': edges_feats,
        }
        return path_feats

    def measures(self, path):
        import operator
        def prod(factors):
            return reduce(operator.mul, factors, 1)

        pattern = {
            'throughput': min,
            'latency': sum,
            'availability': prod,
            'frame_loss_ratio': prod,
        }
        # print 'path', path
        features = pattern.keys()
        path_feats = []
        for id in range(len(path[:-1])):
            src,dst = path[id],path[id+1]
            edge_feats = self.measures_edge((src,dst), features)
            path_feats.append(edge_feats)
        # print path_feats
        edges_feats = self.qualify_edges(path_feats, pattern)

        hops = len(set(path))-1
        edges_feats['hops'] = hops
        return edges_feats

    def qualify_feat(self, path_feats, feat, func):
        feats = [path[feat] for path in path_feats]
        return func(feats)

    def qualify_edges(self, path_feats, pattern):
        feats = {}
        for feat in pattern.keys():
            func = pattern[feat]
            qf = self.qualify_feat(path_feats, feat, func)
            feats[feat] = qf
        return feats

    def measures_edge(self, edge, features):
        (src,dst) = edge
        edge_features = {}
        if src == dst:
            default = {
                'throughput': 1000,
                'latency': 0.0,
                'availability': 1.0,
                'frame_loss_ratio': 1,
            }
            return default

        data = self.g.edge[src][dst]
        # print data
        for feat in features:
            feat_data = data.get(feat, None)
            if feat_data:
                edge_features[feat] = feat_data
        return edge_features

    def measures_node(self, node):
        features = ['betweenness', 'centrality',
                    'vitality', 'clustering', 'avg_neigh_degree',
                    'num_cpus', 'disk_size', 'mem_size']

        node_features = {}
        data = self.g.node[node]
        for feat in features:
            feat_data = data.get(feat, 0)
            node_features[feat] = feat_data
        return node_features


class Classify:
    def __init__(self):
        pass

    def classify(self, service, paths, path_maps):
        costs = {}
        # for cand,features in paths.items():
        #     print cand,features
        for cand,features in paths.items():
            nodes = features.get('nodes')
            edges = features.get('edges')
            joint_costs = self.costs(cand, service, nodes, edges, path_maps)
            # joint_costs is a list of all candidate paths factored (all vs. all options) nf mappings in a chain
            if joint_costs:
                costs[cand] = joint_costs
        return costs

    def service_nodes_resources(self, candidate, nodes, service):
        res = []
        for nf in service:
            # caps = nf.values()[0]
            # host = caps.get('host')
            res.append(nf)

        ordered_node_values = []
        for cand in candidate[1:-1]:
            node_values = nodes[cand]
            ordered_node_values.append(node_values)

        nodes_service_map = zip(ordered_node_values, res)
        return nodes_service_map

    def refine_costs(self, node_costs):
        if node_costs['density'] <= 0 or node_costs['robustness']  <= 0 \
            or node_costs['vitality'] <= 0 or node_costs['reachability'] <= 0 \
            or node_costs['performance'] <= 0:
            return False
        return True

    def costs(self, cand, service, nodes, edges, path_maps):
        overall_costs = []
        map_node_service_resources = self.service_nodes_resources(cand, nodes, service)
        # print cand
        # print edges
        classified_nodes = self.classify_nodes(cand, nodes, map_node_service_resources, path_maps)
        # print 'nodes_costs', nodes_costs
        
        for (nodes_costs, add_edges) in classified_nodes:
            edges_extra = {}
            edges_extra.update(edges)
            edges_extra.update(add_edges)
            edge_costs = self.classify_edges(edges_extra)
            # print 'edge_costs',edge_costs
            costs = {}
            costs.update(nodes_costs)
            costs.update(edge_costs)
            overall_costs.append(costs)

        if classified_nodes:
            return overall_costs
        else:
            return None

        # if self.refine_costs(nodes_costs):
        #     return costs
        # else:
        #     return None

    def classify_nodes_add_edges(self, node_service_map):
        added_edges = {}
        id = 0
        # print "node_service_map", node_service_map
        for (node, req_node) in node_service_map:
            nf_net_profile = req_node.get('network')
            nf_edge = {
                'availability': 1.0,
                'hops': 0,
            }

            nf_edge.update(nf_net_profile)
            added_edges[(id,id)] = nf_edge
            id+=1
        return added_edges

    def decompose_node_service_maps(self, node_service_map):
        map_decomposed = []
        node_maps = {}
        node_id = 0
        for (node, req_node) in node_service_map:
            node_maps[node_id] = []
            for _, properties in req_node.items():
                 node_maps[node_id].append( (node,properties) )      
            node_id += 1

        node_maps_lists =  node_maps.values()
        
        # if len(node_maps) > 1:
        map_decomposed = list(itertools.product(*node_maps_lists))
        # else:
        #     map_decomposed = node_maps_lists
        return map_decomposed

    def classify_nodes(self, cand, nodes, map_node_service_resources, path_maps):
        overall_costs = []
        
        decomposed_node_service_maps = self.decompose_node_service_maps(map_node_service_resources)
        # print "len", len(decomposed_node_service_maps)
        for map_node_service_resource in decomposed_node_service_maps:
            # resources_weights = {'num_cpus': 0.33, 'mem_size': 0.33, 'disk_size': 0.33}
            nodes_costs, added_edges = {}, {}
            resources_diff, resources_footprint = self.diff_nodes_resources(cand, nodes, map_node_service_resource)
            if resources_diff:
                added_edges = self.classify_nodes_add_edges(map_node_service_resource)
                
                topo_metrics = self.classify_nodes_topo(nodes, cand, path_maps)
                resources_diff_sum = self.sum_nodes_resources(resources_footprint)
                
                nodes_costs.update(resources_diff_sum)
                nodes_costs.update(topo_metrics)
                overall_costs.append( (nodes_costs, added_edges) )
        return overall_costs

    def sum_nodes_resources(self, nodes_diffs):
        sums = nodes_diffs.pop()
        for diff in nodes_diffs:
            for feat in diff:
                sums[feat]  += diff[feat]         
        return sums

    def select_node_host(self, node, req_node):
        ack = all( [node[feat] >= req_node[feat] for feat in req_node] )
        return ack

    def diff_nodes_resources(self, cand, nodes, node_service_map):
        def diff_dicts(one, another):
            diffs = {}
            #another contains least features - fix: change for features_set
            for feat in another:
                diffs[feat] = one[feat] - another[feat]
            return diffs

        # print "cand", cand
        nodes_diffs = []
        nodes_footprint = []
        cand_nodes = cand[1:-1]
        common_cands = {}
        ind = 0

        # print cand_nodes
        # print "len node_service_map", len(node_service_map)
        for (node, req_node) in node_service_map:
            # print 'node, req_node', node, req_node
            diff_res = {}
            cand = cand_nodes[ind]
            if cand not in common_cands:
                # profile, cand_profile = self.select_node_host(req_node, node)
                ack = self.select_node_host(node, req_node.get('host'))
                if ack:
                    # selected_nf_profiles[ind] = profile
                    diff_res = diff_dicts(node, req_node.get('host'))                
            else:
                common_node = common_cands[cand]
                ack = self.select_node_host(node, req_node.get('host'))
                if ack:
                    # selected_nf_profiles[ind] = profile
                    diff_res = diff_dicts(common_node, req_node.get('host'))
                    
            # print 'diff_res', diff_res
            if diff_res:
                cost = sum( [req_node.get("cost") for (node, req_node) in node_service_map])
                common_cands[cand] = diff_res
                
                node_footprint = {}
                node_footprint.update(req_node.get('host'))
                node_footprint['cost'] = cost                
                nodes_footprint.append(node_footprint)

                nodes_diff = {}
                nodes_diff.update(diff_res)
                nodes_diff['cost'] = cost
                nodes_diffs.append(nodes_diff)
            else:
                nodes_diffs = []
                break
            ind += 1
        return nodes_diffs, nodes_footprint

    def classify_nodes_topo(self, nodes, cand, path_maps):
        features = {'vitality':'vitality','robustness':'betweenness',
                    'reachability':'centrality',  'density':'avg_neigh_degree'}
        topo_feats = {'vitality': 0.0, 'robustness': 0.0,
                      'reachability': 0.0, 'density': 0.0}

        # mapped_nodes = nodes.values()[1:-1]
        # print mapped_nodes
        # for node_feat in mapped_nodes:
        #     for cls,feat in features.items():
        #         topo_feats[cls] += node_feat[feat]

        mapped_adjs = path_maps[cand[1:-1]]
        for node_id,node_feat in mapped_adjs.items():
            # print node_id,node_feat
            for cls,feat in features.items():
                # node_feats_ana = node_feat.get('capabilities').get('analysis')
                topo_feats[cls] += node_feat[feat]
                # topo_feats[cls] += node_feat[feat]
        return topo_feats

    def classify_edges(self, edges):
        def qualify_feat(path_feats, feat, func):
            feats = [path[feat] for path in path_feats]
            return func(feats)

        import operator
        def prod(factors):
            return reduce(operator.mul, factors, 1)

        pattern = {
            'throughput': min,
            'latency': sum,
            'availability': prod,
            'frame_loss_ratio': prod,
            'hops': sum,
        }
        edges_feats = edges.values()

        feats = {}
        for feat in pattern.keys():
            func = pattern[feat]
            qf = qualify_feat(edges_feats, feat, func)
            feats[feat] = qf
        return feats

    def flexibility(self, paths):
        df = pd.DataFrame(data=paths)
        l,c = df.shape
        # print df
        freqs = []
        for i in range(c):
            freq = df[i].value_counts()
            total = freq.sum()
            freqs.append(freq.apply(lambda s: s/float(total)))

        paths_flex = {}
        for p in paths:
            paths_flex[p] = []
            for i in range(c):
                v = p[i]
                vf = freqs[i][v]
                paths_flex[p].append(vf)
        # print paths_flex
        return paths_flex


class Scores:
    IMAGES_PATH = './images/'

    def __init__(self):
        self.data = pd.DataFrame()
        self.folder = None
        self.prefix = ''

    def set(self, folder, prefix):
        self.folder = folder
        self.prefix = prefix
        self.images_path = self.IMAGES_PATH + self.folder + '/'

        if not os.path.exists(os.path.dirname(self.images_path)):
            os.makedirs(os.path.dirname(self.images_path))

    def get(self, data):
        self.data = pd.DataFrame(data.values())

    def head(self):
        print(self.data.head())

    def info(self):
        print(self.data.info())

    def describe(self):
        print(self.data.describe())

    def save_fig(self, fig_id, tight_layout=True, fig_extension="pdf", fig_size=(8, 6), resolution=500):
        path = os.path.join(self.images_path, fig_id + "." + fig_extension)
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, figsize=fig_size, dpi=resolution)
        self.finish()

    def hists(self):
        filename = self.prefix + "data_histogram_plots"
        self.data.hist(bins=100, figsize=(20, 15))
        self.save_fig(filename)
        # plt.show()

    def hist(self):
        filename = self.prefix + "data_histogram_score"
        self.data.hist(bins=100, column='score', figsize=(20, 15))
        plt.xlim([.5, 1.0])
        self.save_fig(filename)

    def scatter(self):
        filename = self.prefix + "data_scatterplot"
        self.data.plot(kind="scatter", x="hops", y="robustness", alpha=0.4,
              s=self.data["performance"] , label="performance", figsize=(10, 7),
              c="score", cmap=plt.get_cmap("jet"), colorbar=True,
              sharex=False)
        plt.legend()
        self.save_fig(filename)

    def scatter_01(self):
        filename = self.prefix + "data_scatterplot_01"
        self.data.plot(kind="scatter", x="reachability", y="vitality", alpha=0.4,
                            s=self.data["performance"], label="performance", figsize=(10, 7),
                            c="score", cmap=plt.get_cmap("jet"), colorbar=True,
                            sharex=False)
        plt.legend()
        self.save_fig(filename)

    def scatter_02(self):
        filename = self.prefix + "data_scatterplot_02"
        self.data.plot(kind="scatter", x="hops", y="score", alpha=0.4,
                            s=self.data["performance"], label="performance", figsize=(10, 7),
                            c="latency", cmap=plt.get_cmap("jet"), colorbar=True,
                            sharex=False)
        plt.legend()
        self.save_fig(filename)

    def scatter_03(self):
        filename = self.prefix + "data_scatterplot_03"
        self.data.plot(kind="scatter", x="latency", y="score", alpha=0.4,
                            s=self.data["performance"], label="performance", figsize=(10, 7),
                            c="reachability", cmap=plt.get_cmap("jet"), colorbar=True,
                            sharex=False)
        plt.legend()
        self.save_fig(filename)


    def scatter_04(self):
        filename = self.prefix + "data_scatterplot_04"
        self.data.plot(kind="scatter", x="latency", y="score", alpha=0.4,
                            s=self.data["performance"], label="performance", figsize=(10, 7),
                            c="vitality", cmap=plt.get_cmap("jet"), colorbar=True,
                            sharex=False)
        plt.legend()
        self.save_fig(filename)

    def corr(self, feat):
        corr_matrix = self.data.corr()
        print(corr_matrix[feat].sort_values(ascending=False))

    def heatmap(self):
        filename = self.prefix + "data_heatmap"
        # ax = sns.heatmap(self.data, cmap='RdYlGn_r')
        df_norm = (self.data - self.data.mean()) / (self.data.max() - self.data.min())
        df_norm.replace(np.inf, np.nan)
        df_norm.fillna(value=1.0, inplace=True)

        g = sns.clustermap(df_norm, method='centroid', metric='euclidean')

        # turn the axis label
        # for item in ax.get_yticklabels():
        #     item.set_rotation(0)
        #
        # for item in ax.get_xticklabels():
        #     item.set_rotation(45)

        # plt.xticks(rotation=45)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45)

        self.save_fig(filename, fig_size=(13,13), tight_layout=False)

    def heatmap_data(self, input_data):
        filename = self.prefix + "data_heatmap_data"

        data = pd.DataFrame(input_data)
        # ax = sns.heatmap(self.data, cmap='RdYlGn_r')
        df_norm = (data - data.min()) / (data.max() - data.min())
        df_norm.replace(np.inf, np.nan)
        # print df_norm
        df_norm.fillna(value=1.0, inplace=True)
        # print df_norm
        sns.clustermap(df_norm)

        # turn the axis label
        # for item in ax.get_yticklabels():
        #     item.set_rotation(0)
        #
        # for item in ax.get_xticklabels():
        #     item.set_rotation(90)
        self.save_fig(filename, tight_layout=False)

    def boxplot(self):
        filename = self.prefix + "data_boxplot"
        # ax = sns.heatmap(self.data, cmap='RdYlGn_r')
        plt.figure(figsize=(10, 6))
        df_norm = (self.data - self.data.mean()) / (self.data.max() - self.data.min())
        ax = sns.boxplot(data=df_norm)

        # turn the axis label
        # for item in ax.get_yticklabels():
        #     item.set_rotation(0)
        #

        # ax = sns.violinplot(data=df_norm, inner=None)
        # ax = sns.swarmplot(data=df_norm,
        #               alpha=0.7,
        #               color="white", edgecolor="gray")
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        self.save_fig(filename)

    def swarmplot(self, feat1, feat2, hue=None):
        filename = self.prefix + 'data_swarmplot'
        sns.swarmplot(x=feat1, y=feat2, data=self.data, hue=hue)
        self.save_fig(filename)

    def finish(self):
        plt.cla()
        plt.clf()
        plt.close()

    def disthist(self, feat):
        filename = self.prefix + 'data_disthist'
        sns.distplot(self.data[feat], rug=True, norm_hist=True)
        self.save_fig(filename)

    def factorplot(self):
        filename = self.prefix + 'data_factor'

        g = sns.factorplot(x='hops',
                       y='score',
                       data=self.data,
                       hue='latency',  # Color by stage
                       col='throughput',  # Separate by stage
                       kind='swarm',
                       size=8,
                       aspect=1.2,
                       legend_out=True)  # Swarmplot

        # box = g.ax.get_position()  # get position of figure
        # g.ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
        # # Put a legend to the right side
        # g.ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)

        # plt.legend(bbox_to_anchor=(1, 1), loc=2)
        # ax.legend(ncol=1, loc="lower right", frameon=False)
        # ax.despine(left=True)
        self.save_fig(filename)

    def jointplot(self, feat1, feat2, input_data=None):
        filename = self.prefix + 'data_joint'
        if input_data:
            data = pd.DataFrame(input_data)
        else:
            data = self.data
        sns.jointplot(x=feat1, y=feat2, data=data, kind="reg",
                      marginal_kws = dict(bins=15, rug=True), annot_kws = dict(stat="r"))
        self.save_fig(filename)


class Rank:
    def __init__(self):
        self.ahp = AHP()
        self.topsys = TOPSIS()
        self.feats = []
        self.ranking = []

    def rank_topsys(self):
        self.topsys.normalizeMatrix()
        self.topsys.introWeights()
        self.topsys.getIdealSolutions()
        self.topsys.distanceToIdeal()
        self.ranking = self.topsys.relativeCloseness()

    def rank_ahp(self):
        self.ranking = self.ahp.ahp()

    def rank(self, method='topsys'):
        if method == 'topsys':
            self.rank_topsys()
        elif method == 'ahp':
            self.rank_ahp()

        return self.ranking

    def set_ahp(self, costs, preferences, data):
        self.ahp.set(data, costs=costs, preferences=preferences)

    def set_topsys(self, weights, costs, data):
        self.topsys.set_costs(costs)
        self.topsys.set_data(data)
        self.topsys.set_weights(weights)



if __name__ == '__main__':
    # metric = 'latency'
    #
    # ana = Analysis()
    # ana.structure(1)
    # # g = ana.get_graph()
    # # ana.topos.topos.to_raw(g)
    # # ana.topos.save_graph(g, 'slice-01')
    # g = ana.topos.retrieve_graph('slice-01')
    # ana.set_graph(g)
    # print 'is_connected', netx.is_connected(g)
    #
    # # ana.topos.inspect(g)
    # # ana.topos.draw_graph(g)
    #
    # service_policy = {
    #     'affinity':[],
    #     'anti-affinity':[],
    #     'paths': {}
    # }
    #
    # service_chain = [ {'nf1':{'epa':['cpu_pinning'],
    #                           'host':{'num_cpus':2, 'mem_size':4, 'disk_size':10}
    #                           }},
    #                   {'nf2':{'epa':['IO_pass_through'],
    #                           'host': {'num_cpus': 2, 'mem_size': 4, 'disk_size': 10}
    #                           }},
    #                   {'nf3':{'epa':['NUMA'],
    #                           'host': {'num_cpus': 2, 'mem_size': 4, 'disk_size': 10}
    #                           }} ]
    #
    # candidate_chains = ana.topos.candidate_paths(g, service_chain)
    # print candidate_chains
    #
    # ana.augment(g, metric)
    # print 'shortcuts added', len(ana.shortcuts)
    # ana.set_graph(g)
    #
    # paths = ana.connect(g, candidate_chains, metric)
    #
    # # print 'is_connected', netx.is_connected(g)
    # print 'paths', len(paths)
    #
    # # for candidate,value in paths.items():
    # #     dists,path = value
    # #     print candidate
    # #     ana.qualify(path, metric)
    #
    # path_feats = ana.qualify_paths( paths )
    # for candidate,value in path_feats.items():
    #     print candidate, value
    #
    # cfy = Classify()
    # classified = cfy.classify(service_chain, path_feats)
    # print classified
    #
    # path_set = classified.keys()
    # path_feats = classified.values()
    #
    # weights = {'latency': 0.1, 'density': 0.1, 'hops': 0.1, 'frame_loss_ratio': 0.1,
    #            'robustness': 0.1, 'reachability': 0.1, 'throughput': 0.1,
    #            'performance': 0.1, 'availability': 0.1, 'vitality': 0.1}
    #
    # costs = {'latency': 1, 'density': 0, 'hops': 1, 'frame_loss_ratio': 1,
    #          'robustness': 0, 'reachability': 0, 'throughput': 0,
    #          'performance': 1, 'availability': 0, 'vitality': 0}
    #
    # print weights.keys()
    #
    # print path_feats[0].keys()
    # print path_feats[0].values()
    # print path_feats[0]
    #
    # weights_values = []
    # for k in path_feats[0].keys():
    #     weights_values.append(weights[k])
    #
    # costs_values = []
    # for k in path_feats[0].keys():
    #     costs_values.append(costs[k])
    #
    # path_data = [ pf.values() for pf in path_feats]
    #
    # rank = Rank()
    # rank.set_topsys(weights_values, costs_values, path_data)
    # ranking = rank.rank()
    # print len(ranking)
    # print ranking
    #
    # ind = 0
    # for value in classified.values():
    #     score = ranking[ind]
    #     value['score'] = score
    #     ind += 1
    #
    # score = Scores()
    # score.get(classified)
    # # score.head()
    # # score.info()
    # # score.hists()
    # # score.scatter()
    # # score.corr('score')
    # # score.heatmap()
    # # score.boxplot()
    # # score.describe()
    # # score.swarmplot('latency', 'score', hue='throughput')
    # # score.disthist('score')
    # # score.factorplot()
    # # score.jointplot('robustness', 'density')
    #
    # print 'top 10'
    # values = []
    # keys = []
    # classified_ord = sorted(classified.items(),
    #                         key=lambda (k,v): v['score'],
    #                         reverse=True)
    #
    # top_classified_ord = classified_ord[:50]
    # for k,v in top_classified_ord:
    #     values.append(v)
    #     keys.append(k)
    #
    # def pair_path(path):
    #     pairs = []
    #     for i in range(len(path)-1):
    #         s = path[i]
    #         d = path[i+1]
    #         pairs.append( (s,d) )
    #     return (tuple(pairs), path)
    #
    # pair_keys = dict(map(pair_path, keys))
    #
    # score.heatmap_data(values)
    #
    #
    # nodes_flex = cfy.flexibility(keys)
    # edges_flex = cfy.flexibility(pair_keys.keys())
    #
    #
    # links_flex = {}
    # for k,v in edges_flex.items():
    #     nk = pair_keys[k]
    #     links_flex[nk] = v
    #
    # flexs = {}
    #
    # for k in nodes_flex.keys():
    #     nodes_prod = np.prod(np.array(nodes_flex[k]))
    #     links_prod = np.prod(np.array(links_flex[k]))
    #     flexs[k] = nodes_prod + links_prod
    #
    # top_classified_ord_dict = dict(top_classified_ord)
    # print flexs
    # for k in flexs:
    #     vf = flexs[k]
    #     top_classified_ord_dict[k]['flexibility'] = vf
    #
    # classified_flex_ord = sorted(top_classified_ord_dict.items(),
    #                              key=lambda (k, v): v['flexibility'],
    #                              reverse=False)
    #
    # print top_classified_ord
    # print classified_flex_ord
    # top_flex_keys = []
    # top_flex_values = []
    # for k,v in classified_flex_ord:
    #     top_flex_keys.append(k)
    #     top_flex_values.append(v)
    #     print v['flexibility']
    #
    # print top_flex_keys
    # print keys
    #
    # print values
    # print top_flex_values
    #
    # features_top = top_flex_values[0].keys()
    # score.radarplot(features_top,values)
    #
    # # score.heatmap_data(top_flex_values)
    # # score.jointplot('flexibility', 'score', input_data=top_flex_values)
    # # path_names = [ str(num) for num in range(len(path_set))]
    # # rank.set_features(path_names)
    # # rank.plot()
    pass
    # topos = Topologies()
    # for model in range(1,5):
    #     topos.create_model(model, str(model))
