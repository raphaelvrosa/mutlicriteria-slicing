import os
import json
import numpy as np
import itertools
from collections import namedtuple
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from main.slicing import Classify, Analysis, Rank, Scores, Topologies
from graphs.graph_rand import EPACapability, HostCapability, FlowCapability


class VNFProfiles:    
    CPU_RANGE = np.linspace(2, 16, 8)
    MEM_RANGE = np.linspace(2, 16, 8)
            
    def __init__(self):
        self.catalogue = {}

    def get(self, profile_id=0):
        if profile_id == 0:
            return self.profile_00()
        elif profile_id == 1:
            return self.profile_01()
        elif profile_id == 2:
            return self.profile_02()
        elif profile_id == 3:
            return self.profile_03()
        else:
            return self.profile_00()

    def profile_00(self):
        throughput = [20, 40, 60, 80, 100, 120, 140, 160]
        latency = [20, 16, 14, 12, 10, 8, 6, 4]
        cost = [0.1, 0.12, 0.14, 0.16, 0.2, 0.4, 0.6, 0.8]

        profile = {}
        for i in range(len(throughput)):
            profile[i] = {
                'host': {'num_cpus':self.CPU_RANGE[i], 'mem_size':self.MEM_RANGE[i], 'disk_size':10},
                'network': {'throughput': throughput[i], 'latency': latency[i], 'frame_loss_ratio': 1},
                'cost': cost[i],
            }        
        return profile

    def profile_01(self):
        throughput = [20, 30, 40, 50, 50, 50, 50, 50]
        latency = [20, 16, 14, 12, 12, 12, 12, 12]
        cost = [0.1, 0.12, 0.14, 0.16, 0.2, 0.4, 0.6, 0.8]

        profile = {}
        for i in range(len(throughput)):
            profile[i] = {
                'host': {'num_cpus':self.CPU_RANGE[i], 'mem_size':self.MEM_RANGE[i], 'disk_size':10},
                'network': {'throughput': throughput[i], 'latency': latency[i], 'frame_loss_ratio': 1},
                'cost': cost[i],
            }        
        return profile

    def profile_02(self):
        throughput = [20, 30, 40, 50, 60, 70, 80, 90]
        latency = [12, 12, 12, 12, 12, 12, 12, 12]
        cost = [0.1, 0.12, 0.14, 0.16, 0.2, 0.4, 0.6, 0.8]

        profile = {}
        for i in range(len(throughput)):
            profile[i] = {
                'host': {'num_cpus':self.CPU_RANGE[i], 'mem_size':self.MEM_RANGE[i], 'disk_size':10},
                'network': {'throughput': throughput[i], 'latency': latency[i], 'frame_loss_ratio': 1},
                'cost': cost[i],
            }        
        return profile

    def profile_03(self):
        throughput = [50, 50, 50, 50, 50, 50, 50, 50]
        latency = [20, 16, 14, 12, 10, 8, 6, 4]
        cost = [0.1, 0.12, 0.14, 0.16, 0.2, 0.4, 0.6, 0.8]

        profile = {}
        for i in range(len(throughput)):
            profile[i] = {
                'host': {'num_cpus':self.CPU_RANGE[i], 'mem_size':self.MEM_RANGE[i], 'disk_size':10},
                'network': {'throughput': throughput[i], 'latency': latency[i], 'frame_loss_ratio': 1},
                'cost': cost[i],
            }        
        return profile


class ChainProfiles:
    def __init__(self):
        self.nfs = []
        self.vnf_profiles = VNFProfiles()

    def parse_profiles(self, service_nfs, profiles):
        if type(profiles) is int:
            list_profiles = [profiles]*service_nfs
        elif type(profiles) is list:
            assert(len(profiles)==service_nfs)
            list_profiles = profiles
        else:
            list_profiles = [0]*service_nfs #default nf profile id=0
        return list_profiles

    def create(self, service_nfs=3, profiles=None):
        self.nfs = []
        list_profiles = self.parse_profiles(service_nfs, profiles)
        for _id in range(service_nfs):
            profile_id = list_profiles[_id]
            nf = self.vnf_profiles.get(profile_id=profile_id)
            self.nfs.append(nf)
        return self.nfs


class Services:
    def __init__(self, nfs_profile=1, service_capabs=4):
        self.nfs = []
        self.epa = EPACapability(service_capabs)
        self.host = HostCapability(nfs_profile, profile_nf=True)
        self.flow = FlowCapability(nfs_profile)

    def profiles(self, choice):
        profiles = {
            'nf1': {
                'low': {'host': {'num_cpus':2, 'mem_size':4, 'disk_size':10,},
                        'network': {'throughput': 1, 'latency': 16, 'frame_loss_ratio': .8} },
                'medium': {'host': {'num_cpus': 4, 'mem_size': 8, 'disk_size': 50,},
                        'network': {'throughput': 2, 'latency': 8, 'frame_loss_ratio': .4}},
                'high': {'host': {'num_cpus': 8, 'mem_size': 16, 'disk_size': 100,},
                        'network': {'throughput': 4, 'latency': 4, 'frame_loss_ratio': .2}}
            },
            'nf2': {
                'low': {'host': {'num_cpus':2, 'mem_size':4, 'disk_size':10,},
                        'network': {'throughput': 2, 'latency': 8, 'frame_loss_ratio': .4}},
                'medium': {'host': {'num_cpus': 4, 'mem_size': 8, 'disk_size': 50,},
                        'network': {'throughput': 4, 'latency': 4, 'frame_loss_ratio': .2}},
                'high': {'host': {'num_cpus': 8, 'mem_size': 16, 'disk_size': 100,},
                         'network': {'throughput': 8, 'latency': 2, 'frame_loss_ratio': .1}}
            },
            'nf3': {
                'low': {'host': {'num_cpus':2, 'mem_size':4, 'disk_size':10,},
                         'network': {'throughput': 4, 'latency': 4, 'frame_loss_ratio': .2}},
                'medium': {'host': {'num_cpus': 4, 'mem_size': 8, 'disk_size': 50,},
                         'network': {'throughput': 8, 'latency': 2, 'frame_loss_ratio': .1}},
                'high': {'host': {'num_cpus': 8, 'mem_size': 16, 'disk_size': 100,},
                         'network': {'throughput': 16, 'latency': 1, 'frame_loss_ratio': 0.05}}
            },
        }
        if choice == 1:
            chain = [profiles['nf1'], profiles['nf1'], profiles['nf1']]
        elif choice == 2:
            chain = [profiles['nf2'], profiles['nf2'], profiles['nf2']]
        elif choice == 3:
            chain = [profiles['nf3'], profiles['nf3'], profiles['nf3']]
        else:
            chain = [profiles['nf1'], profiles['nf2'], profiles['nf3']]

        return chain

    def create(self, service_profile=1, service_nfs=2, service_capabs=2, enable_range=True):
        # self.host = HostCapability(service_profile, profile_nf=True)
        self.nfs = []
        ids = []
        chain = self.profiles(service_profile)
        for _id in range(service_nfs):
            ids.append(_id)
            nf = {
                    'id':_id,
                    'epa': self.epa.get(profile=service_capabs, alloc=True),
                    # 'host': self.host.get(range=enable_range, suffix=False),
                    'host': chain[_id]
            }
            self.nfs.append(nf)
        print self.nfs
        return self.nfs


class Scenario:

    def __init__(self):
        self.topo = None
        self.service = None
        self.profiles = ChainProfiles()
        self.services = Services()
        self.topologies = Topologies()
        self.scores = []
        # self.weights = {'latency': 0.1, 'density': 0.1, 'hops': 0.1, 'frame_loss_ratio': 0.1,
        #            'robustness': 0.1, 'reachability': 0.1, 'throughput': 0.1,
        #            'performance': 0.1, 'availability': 0.1, 'vitality': 0.1}

        self.weights = {'latency': 0.3, 'density': 0.05, 'hops': 0.05, 'frame_loss_ratio': 0.05,
                   'robustness': 0.05, 'reachability': 0.05, 'throughput': 0.05,
                   'performance': 0.3, 'availability': 0.05, 'vitality': 0.05}

        self.costs = {'latency': 1, 'density': 0, 'hops': 1, 'frame_loss_ratio': 1,
                 'robustness': 0, 'reachability': 0, 'throughput': 0,
                 'performance': 1, 'availability': 0, 'vitality': 0}

        self.weights_pref = [ [1, ] ]

        self.preferences = []
        self.metric = None
        self.debug = False
        self.folder = '.'
        self.trial = {}

    def build_preferences(self):
        pass

    def update_weights(self, dist):
        pass

    def update_costs(self, mods):
        pass

    def load_topo(self, filename):
        print 'loading ', filename
        self.topo = self.topologies.load(filename)
        # for srcid, dstid, data in self.topo.edges_iter(data=True):
        #     print srcid, dstid, data

    def build_service(self, service_profile):
        # self.service = self.services.create(**service_profile)
        #service_profile = ['service_profile', 'service_nfs', 'service_capabs', 'enable_range']
        service_nfs = service_profile.get("service_nfs")
        service_profile = service_profile.get("service_profile")
        self.service = self.profiles.create(service_nfs=service_nfs, profiles=service_profile)
        # print self.service

    def build(self, profile):
        self.trial = profile
        self.metric = profile.get('metric')
        self.debug = profile.get('debug')
        self.folder = profile.get('folder')

        infra_topo_profile = profile.get('topology_profile')
        service_profile = profile.get('service_profile')
        model = infra_topo_profile.get('model')
        size = infra_topo_profile.get('size')
        infra_profile = infra_topo_profile.get('infra_profile')
        infra_nodes_rate = infra_topo_profile.get('infra_nodes_rate')
        infra_mode = infra_topo_profile.get('infra_mode')

        topo_name = 'topo_' + str(size) + '_' + str(model) + '_' + str(infra_profile) + '_' + str(infra_nodes_rate) + '_' + infra_mode
        self.topo = topo_name
        # self.load_topo(topo_name)
        self.build_service(service_profile)
        return True

    def verify(self, profile):
        service_profile = profile.get('service_profile')
        check_till_ok = self.check_build(self.topo, self.service)
        trials_max = 5
        trials = 0
        while not check_till_ok:
            self.build_service(service_profile)
            check_till_ok = self.check_build(self.topo, self.service)
            trials += 1
            if trials >= trials_max:
                return False
        return True

    def check_build(self, topo, service):
        candidate_chains = self.topologies.candidate_paths(topo, service)
        if candidate_chains:
            print 'candidate chains found OK', 'Total #paths', len(candidate_chains)
            return True
        return False

    def get_profile(self):
        profile = {
            'metric': self.metric,
            'topology': self.topo,
            'service': self.service,
            'weights': self.weights,
            'costs': self.costs,
            'preferences': self.preferences,
            'debug': self.debug,
            'trial': self.trial,
            'folder': self.folder,
        }
        return profile

    def set_scores(self, values):
        self.scores = values

    def save_json(self, data, folder , filename):
        filepath = folder + filename + '.json'
        with open(filepath, 'w') as outfile:
            json.dump(data, outfile, indent=4, sort_keys=True)
            return True

    def save(self, profile, folder, fileid):
        del profile['topology']
        self.save_json(profile, folder, fileid)


class Pipe:
    FOLDER = './data/'

    def __init__(self):
        self.scenario = Scenario()
        self.ana = Analysis()
        self.score = Scores()
        self.cfy = Classify()
        self.rank = Rank()
        self.path_metric = None
        self.scenarios = {}
        self.scenarios_ids = 1

    def config(self, trials):
        ack = True
        for trial in trials:
            if not self.build_trial(trial):
                ack = False
        return ack

    def build_trial(self, trial):
        if self.scenario.build(trial):
            sc_profile = self.scenario.get_profile()
            _id = self.scenarios_ids
            sc_profile['id'] = _id
            self.scenarios[_id] = sc_profile
            self.scenarios_ids += 1
            print 'scenario added ', _id
            return True
        return False

    def run(self):
        total_scs = len(self.scenarios)
        for sc in self.scenarios:
            print 'walkthrough', sc, 'total - ', total_scs
            self.walkthrough(sc)

    def walkthrough(self, sc_id):
        scenario = self.scenarios[sc_id]
        infra_name = scenario.get('topology')
        service_chain = scenario.get('service')
        metric = scenario.get('metric')
        weights = scenario.get('weights')
        costs = scenario.get('costs')
        preferences = scenario.get('preferences')
        debug = scenario.get('debug')
        folder = scenario.get('folder')

        print 'infra_name', infra_name
        path_feats, path_maps = self.paths(infra_name, service_chain, metric, debug=debug)
        if path_feats:
            # print 'total candidsates', len(path_feats)
            # print 'path_feats', path_feats
            
            classified = self.classes(service_chain, path_feats, path_maps, debug=debug)
            # print 'total classified', len(classified)

            path_feats = classified.values()

            # self.ranking(classified, path_feats, weights, costs, preferences, debug=debug)
            plot_folder, prefix = folder + str(sc_id), str(sc_id)

            #TODO uncomment for walkthrough plots
            # self.ploting(plot_folder, prefix, classified, debug=debug)
            # top_classified_flexibility = self.flexibility(classified)
            parsed_classified = self.parse_classified(classified)
            print 'total parsed classified', len(parsed_classified)
            self.save(folder, sc_id, parsed_classified)
            print 'saved scenario'

    def parse_classified(self, classified):
        parsed = {}
        for cand, paths in classified.items():
            ind = 0
            for path in paths:
              parsed[(cand,ind)] = path
              ind += 1  
        return parsed

    def parse_keys(self, data):
        k = data.keys()
        v = data.values()
        k1 = [str(i) for i in k] # load uses eval instead of str
        return dict(zip(*[k1, v]))

    def save(self, folder, sc_id, data):
        folder = self.FOLDER + folder
        if not os.path.exists(os.path.dirname(folder)):
            os.makedirs(os.path.dirname(folder))

        fileid = str(sc_id)
        scenario_profile = self.scenarios.get(sc_id)
        self.scenario.save(scenario_profile, folder, 'profile_'+fileid)

        data_parsed = self.parse_keys(data)
        # top_parsed = self.parse_keys(top)
        self.scenario.save_json(data_parsed, folder, 'data_'+fileid)
        # self.scenario.save_json(top_parsed, folder, 'top_'+fileid)

    def paths(self, infra_name, service_chain, metric, debug=True):
        infra_topo = self.scenario.topologies.load(infra_name)
        candidate_chains = self.scenario.topologies.candidate_paths(infra_topo, service_chain)
        if not candidate_chains:
            return None, None

        # self.ana.augment(infra_topo, metric)
        self.ana.set_graph(infra_topo)

        # if debug:
        #     print 'candidate_chains', len(candidate_chains)
        #     print 'shortcuts added', len(self.ana.shortcuts)
        #     # print 'paths', len(paths)

        paths = self.ana.connect(infra_topo, candidate_chains, metric)
        path_feats = self.ana.qualify_paths(paths)
        path_maps = self.ana.qualify_path_mappings(infra_topo, paths)
        return path_feats, path_maps

    def classes(self, service_chain, path_feats, path_maps, debug=True):
        classified = self.cfy.classify(service_chain, path_feats, path_maps)
        print 'total candidates', len(classified)
        if debug:
            for candidate, value in path_feats.items():
                print candidate, value
            # print classified

        # path_set = classified.keys()
        # path_feats = classified.values()
        return classified

    def ranking(self, classified, path_feats, weights, costs,
                preferences, method='topsys', debug=True):

        weights_values = []
        for k in path_feats[0].keys():
            weights_values.append(weights[k])

        costs_values = []
        for k in path_feats[0].keys():
            costs_values.append(costs[k])

        path_data = [pf.values() for pf in path_feats]

        if method == 'topsys':
            self.rank.set_topsys(weights_values, costs_values, path_data)
        elif method == 'ahp':
            self.rank.set_ahp(costs, preferences, path_data)

        ranking = self.rank.rank(method=method)

        ind = 0
        for value in classified.values():
            score = ranking[ind]
            value['score'] = score
            ind += 1

        print len(ranking)
        print ranking

    def ploting(self, folder, prefix, classified, debug=True):
        self.score.set(folder, prefix)
        self.score.get(classified)
        if debug:
            self.score.head()
            self.score.info()
            self.score.describe()
            self.score.corr('score')

        # self.score.hists()
        # self.score.hist()
        # self.score.scatter()
        # self.score.disthist('score')
        # self.score.jointplot('robustness', 'density')
        
        # self.score.scatter_01()
        # self.score.scatter_02()
        # self.score.scatter_03()
        # self.score.scatter_04()
        
        # self.score.factorplot()
        # self.score.heatmap()
        # self.score.boxplot()
        # self.score.swarmplot('latency', 'score', hue='throughput')

    def flexibility(self, classified, top=50, debug=True):
        values = []
        keys = []
        classified_ord = sorted(classified.items(),
                                key=lambda (k, v): v['score'],
                                reverse=True)

        top_classified_ord = classified_ord[:top]
        for k, v in top_classified_ord:
            values.append(v)
            keys.append(k)

        def pair_path(path):
            pairs = []
            for i in range(len(path) - 1):
                s = path[i]
                d = path[i + 1]
                pairs.append((s, d))
            return (tuple(pairs), path)

        pair_keys = dict(map(pair_path, keys))

        # self.score.heatmap_data(values)

        nodes_flex = self.cfy.flexibility(keys)
        edges_flex = self.cfy.flexibility(pair_keys.keys())

        links_flex = {}
        for k, v in edges_flex.items():
            nk = pair_keys[k]
            links_flex[nk] = v

        flexs = {}
        for k in nodes_flex.keys():
            nodes_prod = np.prod(np.array(nodes_flex[k]))
            links_prod = np.prod(np.array(links_flex[k]))
            flexs[k] = nodes_prod + links_prod

        top_classified_ord_dict = dict(top_classified_ord)
        # print flexs
        for k in flexs:
            vf = flexs[k]
            top_classified_ord_dict[k]['flexibility'] = vf

        classified_flex_ord = sorted(top_classified_ord_dict.items(),
                                     key=lambda (k, v): v['flexibility'],
                                     reverse=False)
        # print top_classified_ord
        # print classified_flex_ord
        top_flex_keys = []
        top_flex_values = []
        for k, v in classified_flex_ord:
            top_flex_keys.append(k)
            top_flex_values.append(v)
        #     print v['flexibility']

        if debug:
            print top_flex_keys
            print keys
            print values
            print top_flex_values

        features_top = top_flex_values[0].keys()
        # self.score.radarplot(features_top, values)

        #TODO uncomment for flexibility walkthrougth plots
        # self.score.jointplot('flexibility', 'score', input_data=top_flex_values)
        # self.score.heatmap_data(top_flex_values)

        return top_classified_ord_dict


class Experiment:

    def __init__(self):
        self.pipe = Pipe()
        self.folder = '0/'
        self.debug = True
        self.keep_topo = False
        self.update_topo = False

        self.trials = []
        self.infra_profile_range = []
        self.infra_profile_range = range(3, 4, 1)
        self.infra_nodes_rate_range = map(lambda x: x / 10.0, range(5, 6, 1))
        self.nets_w_ports_rate_range = map(lambda x: x / 10.0, range(2, 3, 1))
        self.number_enni_ports_range = [1]
        self.enable_range = [True]
        self.model = [1]
        self.infra_mode = ['vitality']
        self.service_nfs_range = range(1, 3, 1)
        self.service_caps_range = range(1, 3, 1)

    def inputs(self, ranges, folder='0/', debug=False, keep_topo=False, update_topo=False):
        self.folder = folder
        self.debug = debug
        self.keep_topo = keep_topo
        self.update_topo = update_topo

        self.size = ranges.size
        self.infra_profile_range = range(*ranges.infra_profile)
        self.infra_nodes_rate_range = map(lambda x: x / 10.0, range(*ranges.infra_nodes_rate))
        self.nets_w_ports_rate_range = map(lambda x: x / 10.0, range(*ranges.nets_w_ports_rate))
        self.number_enni_ports_range = range(*ranges.number_enni_ports)
        self.enable_range = [ranges.enable_range]
        self.model = range(*ranges.model)
        self.infra_mode = ranges.infra_mode
        self.service_profile_range = range(*ranges.service_profile)
        self.service_nfs_range = range(*ranges.service_nfs)
        self.service_caps_range = range(*ranges.service_capabs)

    def create_trials(self):
        topo_features = [
            self.model,
            self.size,
            self.infra_profile_range,
            self.infra_nodes_rate_range,
            self.infra_mode,
            self.nets_w_ports_rate_range,
            self.number_enni_ports_range,
            self.enable_range,
            self.service_profile_range,
            self.service_nfs_range,
            self.service_caps_range,
        ]
        topo_keys = [
            'model',
            'size',
            'infra_profile',
            'infra_nodes_rate',
            'infra_mode',
            'nets_w_ports_rate',
            'number_enni_ports',
            'enable_range'
        ]
        service_keys = ['service_profile', 'service_nfs', 'service_capabs', 'enable_range']
        ord_keys = topo_keys + service_keys
        shuffled_trial_feats = list(itertools.product(*topo_features))

        # print 'shuffled_trial_feats'
        # print shuffled_trial_feats
        for trial_features in shuffled_trial_feats:
            topo_dict = {}
            service_dict = {}
            for i in range(len(trial_features)):
                key = ord_keys[i]
                value = trial_features[i]
                if key in topo_keys:
                    topo_dict[key] = value
                if key in service_keys:
                    service_dict[key] = value

            trial = {
                'debug': self.debug,
                'metric': None, #TODO set paths metric
                'topology_profile': topo_dict,
                'service_profile': service_dict,
                'folder': self.folder,
                'keep_topo': self.keep_topo,
                'update_topo': self.update_topo,
            }
            self.trials.append(trial)

    def build(self):
        all_trials_ok = self.pipe.config(self.trials)
        if all_trials_ok:
            return True
        print 'not all trials built into scenarios'
        # return False
        return True

    def run(self):
        self.pipe.run()


class Outputs:
    DATA_PREFIX = 'data_'
    TOP_PREFIX = 'top_'
    PROFILE_PREFIX = 'profile_'

    def __init__(self):
        self.root_folder = None
        self._files = []
        self.data = {}
        self.profiles = pd.DataFrame()
        self.top = {}
        self.full_data = pd.DataFrame()

    def loading(self, root, file, full_path):
        p = os.path.join(root, file)
        if full_path:
            file_path = os.path.abspath(p)
            self._files.append(file_path)
        else:
            self._files.append(file)

    def load_files(self, folder, file_begin_with, endswith=None, full_path=False):
        self._files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.startswith(file_begin_with):
                    if endswith:
                        if file.endswith(endswith):
                            self.loading(root, file, full_path)
                    else:
                        self.loading(root, file, full_path)
        return self._files

    def load_json(self, filename):
        with open(filename, 'r') as infile:
            data = json.load(infile)
            return data

    def load_top(self, folder):
        files = self.load_files(folder, self.TOP_PREFIX, full_path=True)
        for filename in files:
            fileid = filename.split('_')[-1].split('.')[0]
            data = self.load_json(filename)
            df = pd.DataFrame(data)
            self.top[fileid] = df

    def build_ids(self, files):
        file_ids = {}
        ids = 0
        for filename in sorted(files):
            file_ids[filename] = ids
            ids+=1
        return file_ids

    def load_data(self, folder):
        files = self.load_files(folder, self.DATA_PREFIX, full_path=True)
        filenames = self.build_ids(files)
        for f in filenames:
            # fileid = int(filename.split('_')[-1].split('.')[0])
            fileid = filenames[f]
            data = self.load_json(f)
            df = pd.DataFrame(data)
            self.data[fileid] = df.T

    def load_profiles(self, folder):
        files = self.load_files(folder, self.PROFILE_PREFIX, full_path=True)
        dataframes = []
        indexes = []
        filenames = self.build_ids(files)
        for f in filenames:
            data = self.load_json(f)
            fileid = filenames[f]
            # fileid = data.get('id')
            framedata = data.get("trial")
            framedata.update(framedata.get("service_profile"))
            framedata.update(framedata.get("topology_profile"))
            # del framedata["service_profile"]
            del framedata["topology_profile"]
            indexes.append(fileid)
            dataframes.append(framedata)

        df = pd.DataFrame(index=indexes, data=dataframes)
        self.profiles = df

    def load(self, folder):
        self.root_folder = folder
        self.load_data(folder)
        self.load_profiles(folder)
        # self.load_top(folder)

    def compose_data(self, varying_features):
        service_feats = self.profiles.loc[:, varying_features]
        # print 'service_feats', service_feats
        for _id in self.data:
            df = self.data[_id]
            df_w_index = df.reset_index()
            df_w_index['paths'] = df.index

            for feat in service_feats:
                col_feat = service_feats.loc[_id, feat]
                df_feat = pd.DataFrame([col_feat] * df.shape[0], columns=[feat])
                df_w_index = df_w_index.join(df_feat)

            df_feat = pd.DataFrame([_id] * df.shape[0], columns=["exp_id"])
            df_w_index = df_w_index.join(df_feat)

            self.data[_id] = df_w_index

        composed_df = pd.DataFrame()
        for i in self.data:
            df = self.data[i]
            composed_df = pd.concat([composed_df, df], ignore_index=True)

        self.full_data = composed_df
        # print self.full_data.describe()

    def save(self, filename):
        ext = '.csv'
        filepath = self.root_folder + filename + ext
        self.full_data.to_csv(filepath)

    def save_fig(self, fig_id, tight_layout=True, fig_extension="pdf", fig_size=(8, 6), resolution=500):
        path = os.path.join(self.root_folder, fig_id + "." + fig_extension)
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, figsize=fig_size, dpi=resolution)
        self.finish()

    def finish(self):
        plt.cla()
        plt.clf()
        plt.close()

    def scatter_01(self):
        filename = "exp_scatterplot_01"
        self.full_data.plot(kind="scatter", x="service_nfs", y="score", alpha=0.4,
              s=self.full_data["latency"]*10 , label="latency", figsize=(10, 7),
              c="infra_nodes_rate", cmap=plt.get_cmap("jet"), colorbar=True,
              sharex=False)
        plt.legend()
        self.save_fig(filename)

    def scatter_02(self):
        filename = "exp_scatterplot_02"
        self.full_data.plot(kind="scatter", x="reachability", y="vitality", alpha=0.4,
              s=self.full_data["hops"]*10 , label="hops", figsize=(10, 7),
              c="score", cmap=plt.get_cmap("jet"), colorbar=True,
              sharex=False)
        plt.legend()
        self.save_fig(filename)


    def scatter_03(self):
        filename = "exp_scatterplot_03"
        g = self.full_data.plot(kind="scatter", x="score", y="model", alpha=0.4,
              s=self.full_data["performance"]/100 , label="performance", figsize=(10, 7),
              c="vitality", cmap=plt.get_cmap("jet"), colorbar=True,
              sharex=False)
        plt.legend()
        g.set_yticklabels(['', '1', '',  '2', '', '3', '', '4'])
        self.save_fig(filename)

    def scatter_04(self):
        filename = "exp_scatterplot_04"
        self.full_data.plot(kind="scatter", x="service_nfs", y="service_capabs", alpha=0.4,
              s=self.full_data["hops"]*10 , label="hops", figsize=(10, 7),
              c="score", cmap=plt.get_cmap("jet"), colorbar=True,
              sharex=False)
        plt.legend()
        self.save_fig(filename)

    def scatter_05(self):
        filename = "exp_scatterplot_05"
        g = self.full_data.plot(kind="scatter", x="score", y="model", alpha=0.4,
              s=self.full_data["performance"]/10 , label="performance", figsize=(10, 7),
              c="latency", cmap=plt.get_cmap("jet"), colorbar=True,
              sharex=False)
        plt.legend()
        g.set_yticklabels(['', '1', '', '2', '', '3', '', '4'])
        self.save_fig(filename)


    def factorplot_01(self):
        filename = 'exp_factor_01'
        g = sns.factorplot(x='infra_profile',
                       y='score',
                       data=self.full_data,
                       hue='service_profile',  # Color by stage
                       col='model',  # Separate by stage
                       kind='violin',
                       palette='Set3')  # Swarmplot
        # plt.legend(bbox_to_anchor=(1, 1), loc=2)
        # plt.setp(g.ax_heatmap.get_xticklabels(), xticks=np.arange(1,4,1))
        # g.set(xticklabels=[])
        self.save_fig(filename)


    def factorplot_02(self):
        filename = 'exp_factor_02'
        g = sns.factorplot(x='infra_profile',
                       y='latency',
                       data=self.full_data,
                       hue='service_profile',  # Color by stage
                       col='model',  # Separate by stage
                       kind='violin',
                       palette='Set3',
                       legend=False)  # Swarmplot

        g.despine(left=True)
        plt.legend(title='service_profile', loc='upper right')
        # plt.legend(bbox_to_anchor=(1, 1), loc=2)
        # plt.setp(g.ax_heatmap.get_xticklabels(), xticks=np.arange(1,4,1))
        # g.set(xticklabels=[])
        self.save_fig(filename)


    def factorplot_03(self):
        filename = 'exp_factor_03'
        g = sns.factorplot(x='infra_profile',
                       y='performance',
                       data=self.full_data,
                       hue='service_profile',  # Color by stage
                       col='model',  # Separate by stage
                       kind='violin',
                       palette='Set3',
                       legend=False)  # Swarmplot

        g.despine(left=True)
        plt.legend(title='service_profile', loc='upper right')
        # plt.legend(bbox_to_anchor=(1, 1), loc=2)
        # plt.setp(g.ax_heatmap.get_xticklabels(), xticks=np.arange(1,4,1))
        # g.set(xticklabels=[])
        self.save_fig(filename)


    def factorplot_04(self):
        filename = 'exp_factor_04'
        g = sns.factorplot(x='infra_profile',
                       y='vitality',
                       data=self.full_data,
                       hue='infra_mode',  # Color by stage
                       col='service_profile',  # Separate by stage
                       kind='violin',
                       palette='Set3',
                       legend=False)  # Swarmplot

        g.despine(left=True)
        plt.legend(title='Infra distribution', loc='upper right')
        # plt.legend(bbox_to_anchor=(1, 1), loc=2)
        # plt.setp(g.ax_heatmap.get_xticklabels(), xticks=np.arange(1,4,1))
        # g.set(xticklabels=[])
        self.save_fig(filename)


    def factorplot_05(self):
        filename = 'exp_factor_05'
        g = sns.factorplot(x='infra_profile',
                       y='robustness',
                       data=self.full_data,
                       hue='infra_mode',  # Color by stage
                       col='service_profile',  # Separate by stage
                       kind='violin',
                       palette='Set3',
                       legend=False)  # Swarmplot

        g.despine(left=True)
        plt.legend(title='Infra distribution', loc='upper right')
        # plt.legend(bbox_to_anchor=(1, 1), loc=2)
        # plt.setp(g.ax_heatmap.get_xticklabels(), xticks=np.arange(1,4,1))
        # g.set(xticklabels=[])
        self.save_fig(filename)

    def factorplot_06(self):
        filename = 'exp_factor_06'
        g = sns.factorplot(x='infra_profile',
                       y='score',
                       data=self.full_data,
                       hue='infra_mode',  # Color by stage
                       col='service_profile',  # Separate by stage
                       kind='violin',
                       palette='Set3',
                       legend=False)  # Swarmplot

        g.despine(left=True)
        plt.legend(title='Infra distribution', loc='upper right')
        # plt.legend(bbox_to_anchor=(1, 1), loc=2)
        # plt.setp(g.ax_heatmap.get_xticklabels(), xticks=np.arange(1,4,1))
        # g.set(xticklabels=[])
        self.save_fig(filename)


    def boxplot(self):
        filename = 'exp_boxplot'
        sns.boxplot(x="model", y="score", hue="infra_nodes_rate", data=self.full_data, palette="PRGn")
        sns.despine(offset=10, trim=True)
        self.save_fig(filename)

    def boxplot_01(self):
        filename = 'exp_boxplot_01'
        sns.boxplot(x="service_nfs", y="score", hue="infra_nodes_rate", data=self.full_data, palette="Set3")
        # sns.despine(offset=10, trim=True)
        self.save_fig(filename)

    def boxplot_02(self):
        filename = 'exp_boxplot_02'
        sns.boxplot(x="model", y="score", data=self.full_data, palette="Set3")
        sns.despine(offset=10, trim=True)
        self.save_fig(filename)

    def boxplot_03(self):
        filename = 'exp_boxplot_03'
        sns.boxplot(x="model", y="density", data=self.full_data, palette="Set3")
        sns.despine(offset=10, trim=True)
        self.save_fig(filename)

    def boxplot_04(self):
        filename = 'exp_boxplot_04'
        sns.boxplot(x="model", y="vitality", data=self.full_data, palette="Set3")
        sns.despine(offset=10, trim=True)
        self.save_fig(filename)


    def lmplot(self):
        filename = 'exp_lmplot'
        sns.lmplot(x="infra_nodes_rate", y="service_nfs", hue="score",
                       truncate=True, size=5, data=self.full_data)
        self.save_fig(filename)


    def strip_plot(self):
        filename = 'strip_plot_01'

        # data_plot = pd.melt(self.full_data, "model", var_name="score_model")
        data_plot = self.full_data
        # Initialize the figure
        f, ax = plt.subplots()
        sns.despine(bottom=True, left=True)

        # Show each observation with a scatterplot
        sns.stripplot(x="score", y="model", hue="performance",
                      data=data_plot, dodge=True, jitter=True,
                      alpha=.25, zorder=1)

        # Show the conditional means
        sns.pointplot(x="score", y="model", hue="performance",
                      data=data_plot, dodge=.532, join=False, palette="dark",
                      markers="d", scale=.75, ci=None)

        # Improve the legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[3:], labels[3:], title="performance",
                  handletextpad=0, columnspacing=1,
                  loc="lower right", ncol=3, frameon=True)
        self.save_fig(filename)


    def plots(self):
        #Varying all
        # self.factorplot_01()
        # self.factorplot_02()
        # self.factorplot_03()

        #varying infra mode
        self.factorplot_04()
        self.factorplot_05()
        self.factorplot_06()

        # self.boxplot_01()
        # self.lmplot()
        # self.scatter_01()
        # self.boxplot_02()
        # self.factorplot_03()
        # self.strip_plot()
        # self.boxplot_03()
        # self.boxplot_04()
        # self.scatter_02()
        # self.scatter_03()
        # self.scatter_05()
        # self.scatter_04()
        # self.rand_plot_001()


    def rand_plot_001(self):
        filename = 'exp_randplot_001'
        sns.swarmplot(x="model", y="score", hue="infra_nodes_rate", data=self.full_data)
        self.save_fig(filename)


    def rand_plot_002(self):
        filename = 'exp_randplot_002'
        sns.factorplot(x='infra_nodes_rate',
                       y='score',
                       data=self.full_data,
                       hue='throughput',  # Color by stage
                       col='model',  # Separate by stage
                       kind='box')  # Swarmplot
        # plt.legend(bbox_to_anchor=(1, 1), loc=2)
        self.save_fig(filename)


class Simulation:
    FOLDER = './data/'

    def __init__(self):
        self.root_folder = './'
        self.varying_features = []
        self.trials = 50
        self.exp_template = namedtuple('Experiment', 'model size infra_profile infra_nodes_rate \
                                    infra_mode nets_w_ports_rate number_enni_ports enable_range \
                                    service_profile service_nfs service_capabs')
        self.exp_configs = self.exp_template
        self.params = {}
        self.exps = {}
        self.default_template = self.exp_template(model=[1, 2, 1], size=[10], infra_profile=[3, 4, 1], infra_nodes_rate=[5, 6, 1],
                                        infra_mode=['vitality','centrality','random'], nets_w_ports_rate=[2, 3, 1], number_enni_ports=[1,2,1], enable_range=True,
                                        service_profile=[1, 2, 1], service_nfs=[3,4,1], service_capabs=[4,5,1])

    def configure(self, configs):
        self.params = configs.get('params')
        del configs['params']
        self.exp_configs = self.exp_template(**configs)
        self.root_folder = self.FOLDER + self.params.get('folder')
        self.varying_features = self.params.get('varying_features')
        del self.params['varying_features']

    def update_params_folder(self, trial, params):
        params_folder = {}
        params_folder.update(params)
        params_folder['folder'] = params_folder['folder'] + str(trial) + '/'
        return params_folder

    def build(self, trials=None):
        if trials:
            self.trials = trials
        if not self.exp_configs:
            self.exp_configs = self.default_template

        for trial in range(self.trials):
            exp = Experiment()
            params = self.update_params_folder(trial, self.params)
            exp.inputs(self.exp_configs, **params)
            exp.create_trials()
            build_ok = exp.build()
            if build_ok:
                self.exps[trial] = exp
            else:
                print 'exp trial not built ', trial

    def run(self):
        for trial in self.exps:
            print 'running trial ', trial
            exp = self.exps[trial]
            exp.run()

    def graphics(self, folder=None, varying_features=None):
        if not folder:
            folder = self.root_folder
        if not varying_features:
            varying_features = self.varying_features

        outs = Outputs()
        outs.load(folder)
        outs.compose_data(varying_features)
        # outs.plots()

    def save_data(self, filename, load_folder='./data/', varying_features=None):
        varying_features = self.varying_features if not varying_features else varying_features
        outs = Outputs()
        outs.load(load_folder)
        outs.compose_data(varying_features)
        outs.save(filename)

    def clear(self):
        self.exp_configs = {}
        self.params = {}
        self.exps = {}


if __name__ == '__main__':
    vnfprofs = VNFProfiles()
    # print vnfprofs.profile()
    