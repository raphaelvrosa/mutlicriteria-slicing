import sys
sys.path.append("/home/raphael/PycharmProjects/slices")

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

# create_topos_base()
# create_topos()


chain = ChainProfiles()
nfs = chain.create(service_nfs=5, profiles=None)
for nf in nfs:
    print nf
