from main.pipeline import Simulation
from main.pipeline import ChainProfiles
from main.slicing import Topologies

def create_topos_base():
    nodes_range = [10, 50, 100, 500, 1000]   
    topos = Topologies()

    for nodes in nodes_range: 
        model_kwargs = {'nodes': nodes, 'degree': 5, 'neighbour_edges': 5, 'edge_prob': 0.5}
        for model in range(1, 5):
            for infra_profile in range(1, 5):
                graph_name = str(model)
                graph_size = str(nodes)
                topo_name = 'topo_' + graph_size + '_' + graph_name + '_' + str(infra_profile)
                topo = topos.create_topo_base(model, model_kwargs, infra_profile)
                topos.store(topo, topo_name, base=True)

def create_topos():
    # nodes_range = [10, 50, 100, 500, 1000]
    nodes_range = [10, 50, 100]
    topos = Topologies()
    for nodes in nodes_range: 
        model_kwargs = {'nodes': nodes, 'degree': 5, 'neighbour_edges': 5, 'edge_prob': 0.5}
        for model in range(1, 5):
            for infra_profile in range(1, 5):
                for infra_nodes in range(1, 6, 1):
                    for infra_mode in ['vitality', 'centrality', 'random']:
                        infra_nodes_rate = infra_nodes/10.0
                        graph_name = str(model)
                        graph_size = str(nodes)
                        topo_name = 'topo_' + graph_size + '_' + graph_name + '_' + str(infra_profile) + '_' + str(infra_nodes_rate) + '_' + infra_mode
                        # topo = topos.create_topo(model, model_kwargs, infra_profile, infra_nodes_rate, infra_mode)
                        topo = topos.create_topo_from_raw(model, graph_size, model_kwargs, infra_profile, infra_nodes_rate, infra_mode)
                        topos.store(topo, topo_name)


if __name__ == '__main__':

    # create_topos_base()
    # create_topos()


    prof_full = {
        'size': [10, 50, 100],
        'model': [1, 5, 1],
        'infra_profile': [1, 5, 1],
        'infra_nodes_rate': [2, 3, 2],
        'infra_mode': ['vitality', 'random', 'centrality'],
        'nets_w_ports_rate': [1, 2, 1],
        'number_enni_ports': [1, 2, 1],
        'enable_range': True,
        'service_profile': [0, 4, 1],
        'service_nfs': [1, 4, 1],
        'service_capabs': [3, 4, 1],
        'params': {
            'folder': './', 'debug': False, 'keep_topo': True, 'update_topo': False,
            'varying_features': ['model', 'size', 'infra_nodes_rate', 'infra_mode', 'infra_profile', 'service_profile', 'service_nfs']
        }
    }


    prof_new_var_tst = {
        'size': [100],
        'model': [1, 2, 1],
        'infra_profile': [3, 4, 1],
        'infra_nodes_rate': [3, 4, 2],
        'infra_mode': ['vitality'],
        'nets_w_ports_rate': [1, 2, 1],
        'number_enni_ports': [1, 2, 1],
        'enable_range': True,
        'service_profile': [0, 1, 1],
        'service_nfs': [3, 4, 1],
        'service_capabs': [3, 4, 1],
        'params': {
            'folder': './', 'debug': False, 'keep_topo': True, 'update_topo': False,
            'varying_features': ['model', 'size', 'infra_nodes_rate', 'infra_mode', 'infra_profile', 'service_profile']
        }
    }

    varying_features =  ['model', 'size', 'infra_nodes_rate', 'infra_mode', 'infra_profile', 'service_nfs', 'service_profile']

    sims = Simulation()
    sims.configure(prof_full)
    sims.configure(prof_new_var_tst)
    sims.build(trials=1)
    sims.run()
    sims.save_data('full_data', varying_features=varying_features)
    # sims.graphics()
    # sims.graphics(folder='./exps/exps/test/', varying_features=['model', 'infra_nodes_rate', 'infra_profile'])
