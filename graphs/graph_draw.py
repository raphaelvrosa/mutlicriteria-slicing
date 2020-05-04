import os
import random

import networkx as nx

class GraphView():

    infrastructure = {'NF': False,           # Node
                      'Infra': True,          # Infrastructure Node
                      'Net': True,          # Port of a Node
                      'Port': True,             # Internal link between two ports of a Node
                      'Link': True,            # Internal link between two ports of a Infra Node
                      'Flow': False
                      }     # Edge from a Node's port to a Node (logical)

    allocation = {'NF': True,           # Node
                  'Infra': True,          # Infrastructure Node
                  'Net': True,          # Port of a Node
                  'Port': True,             # Internal link between two ports of a Node
                  'Link': True,            # Internal link between two ports of a Infra Node
                  'Flow': True
                  }     # Edge from a Node's port to a Node (logical)

    higher = {'NF': True,           # Node
                      'Infra': True,          # Infrastructure Node
                      'Net': True,          # Port of a Node
                      'Port': True,             # Internal link between two ports of a Node
                      'Link': True,            # Internal link between two ports of a Infra Node
                      'Flow': True
                      }     # Edge from a Node's port to a Node (logical)


    def __init__(self):
        self.views = dict()
        self._load_default_views()
        self.ignored_colors = ['#000000', '#696969', '#FF0000']

    def _load_default_view_properties(self, view_name):
        labels = {'nodes':{'resources':True, 'ids':True, 'keys':True},
                  'edges':{'resources':True, 'ids':True, 'keys':True}}

        cluster = True
        connected = True
        picture = {'save':True,
                   'size':(20,20),
                   'name':view_name,
                   'ext':'.png'}
        properties = {'labels':labels,
                      'cluster':cluster,
                      'picture':picture,
                      'connected':connected,
                      }

        for k,v in self.__class__.__dict__.items():
            if view_name == k:
                properties['filter'] = v

        return properties

    def _load_default_views(self):
        _default = ["infrastructure",
                    "allocation",
                    "higher"]

        for view_name in _default:
            properties = self._load_default_view_properties(view_name)
            self.add_view(view_name, properties)

    def add_view(self, view_name, properties):
        self.views[view_name] = properties

    def remove_view(self, view_name):
        if view_name in self.views:
            del self.views[view_name]

    def get_view(self, view_name):
        if view_name in self.views:
            return self.views[view_name]
        return None

    def add_view_property(self, view_name, property, value):
        if view_name not in self.views:
            self.views[view_name] = {}
        self.views[view_name][property] = value

    def remove_view_property(self, view_name, property):
        if view_name in self.views:
            if property in self.views[view_name].keys():
                del self.views[view_name][property]

    def apply_view_filter(self, view_name, graph):
        view = self.views[view_name]
        filter_ = view['filter']

        _view_nodes = []
        for node, data in graph.nodes_iter(data=True):
            if 'label' in data:
                if filter_[data['label']]:
                    _view_nodes.append((node, data))
            else:
                raise Exception('"Type" attribute is missing from node '+str(node)+' (data: '+str(repr(data))+')')

        # _view_nodes = filter(lambda node: filter_[node[-1]["type"]],
        #                      graph.nodes_iter(data=True))

        _view_links = filter(lambda edge: filter_[edge[-1]["label"]],
                             graph.edges_iter(keys=True, data=True))
        filter_graph = nx.MultiDiGraph()

        filter_graph.add_nodes_from(_view_nodes)

        try:
            nodes, datas = zip(*_view_nodes)
        except:
            nodes = tuple()
            datas = tuple()

        for link in _view_links:
            if link[1] in nodes:
                filter_graph.add_edge(link[0], link[1], link[2], link[3])

        self.process_view_labels(view_name, filter_graph)
        return filter_graph

    def process_view_labels(self, view_name, graph):
        view = self.views[view_name]
        _labels = view['labels']
        sep = '\n'

        def label_node(node):
            data = node[-1]
            label = ''
            labels = _labels['nodes']
            if data['label'] == 'Infra' or data['label'] == 'NF':
                if labels['ids']:
                    label = sep.join([label,node[0],data['id']])
                if labels['resources'] and 'cpu' in data and 'mem' in data and 'storage' in data and \
                                data['cpu'] is not None and data['cpu'] is not None and data['cpu'] is not None and data['cpu'] is not None:
                    label = sep.join([label, data['cpu'], data['mem'], data['storage']])
                if labels['keys']:
                    label = sep.join([label, data['label']])
            else:
                label = sep.join([label, node[0]])
            data['label'] = label


        def label_edge(edge):
            data = edge[-1]
            label = ''
            labels = _labels['edges']
            if data['label'] != 'INPort2INode' and data['label'] != 'NPort2Node' and data['label'] != 'INode2Node':
                if labels['ids']:
                    label = sep.join([label,edge[0]+'-'+edge[1]])
                if 'delay' in data and 'bandwidth' in data:
                    if labels['resources'] and all(data[res] for res in ['bandwidth','delay']):
                        label = sep.join([label, data['bandwidth'], data['delay']])
                if labels['keys']:
                    label = sep.join(str([label, edge[-2]]))
            data['label'] = label

        map(lambda node: label_node(node),
            graph.nodes_iter(data=True))

        map(lambda edge: label_edge(edge),
            graph.edges_iter(keys=True, data=True))

    def gen_hex_color_code(self):
        return '#' + ''.join([random.choice(['00', '33', '66', '99', 'CC', 'FF']) for x in range(3)])

    def set_color_to_allocations(self, allocations):
        result = {}
        for allocation in allocations:
            color = self.gen_hex_color_code()
            while color in result.keys() or color in self.ignored_colors:
                color = self.gen_hex_color_code()
            result[color] = allocation
        return result


class GraphDraw():
    Infra_size = 1.5
    NF_size = 1.0
    Port_size = 0.6
    Net_size = .5
    Port_Port_len = 4

    infra = {'color': '#3366ff', 'fillcolor': '#3366ff',
            'style': 'filled', 'fontcolor': '#FFFFFF', 'fontsize': 10,
            'width': Infra_size, 'height': Infra_size, 'penwidth': '8.0',
            'fixedsize': True, 'shape': 'square'}

    NF = {'color': '#009900', 'fillcolor': '#009900',
             'style': 'filled', 'fontcolor': '#FFFFFF', 'fontsize': 10,
             'width': NF_size, 'height': NF_size, 'penwidth': '8.0',
             'fixedsize': True, 'shape': 'square',
             'alpha':"0.8"}

    port= {'color': '#3366ff', 'fillcolor': '#3366ff',
            'style': 'filled', 'fontcolor': '#FFFFFF', 'fontsize': 10,
            'width': Port_size, 'height': Port_size, 'penwidth': '8.0',
            'fixedsize': True, 'shape': 'circle'}

    network = {'color': '#009900', 'fillcolor': '#009900',
              'style': 'filled', 'fontcolor': '#FFFFFF', 'fontsize': 10,
              'width': Net_size, 'height': Net_size, 'penwidth': '8.0',
              'fixedsize': True, 'shape': 'circle'}

    link = {'color': '#000000', 'fillcolor': '#000000',
             'style': 'filled', 'fontcolor': '#000000',
             'tickness': '1.0', 'len': '2.0', 'fontsize': 10,
             'fixedsize': True, 'alpha':"0.5"}

    virtuallink = {'color': '#000000', 'fillcolor': '#000000',
             'style': 'filled', 'fontcolor': '#000000',
             'tickness': '1.0', 'len': '2.0', 'fontsize': 10,
             'fixedsize': True, 'alpha':"0.5"}

    flow = {'color': '#000000', 'fillcolor': '#000000',
            'style': 'filled', 'fontcolor': '#000000',
            'tickness': '0.5', 'len': '1.5', 'fontsize': 10,
            'fixedsize': True}

    Port = {'color': '#ffff00', 'fillcolor': '#ffff00',
                    'style': 'filled', 'fontcolor': '#000000',
                    'tickness': '0.5', 'len': '1.5', 'fontsize': 10,
                    'fixedsize': True}

    def __init__(self):
        self.visual_properties = dict()
        self._load_default_properties()

    def _load_default_properties(self):
        _default = ["infra", "NF", "port",
                    "network", "link", "flow",
                    "Port", 'virtuallink']

        for k,v in self.__class__.__dict__.items():
            if k in _default:
                self.add_properties(k,v)

    def reset_visualization(self):
        self._load_default_properties()

    def add_properties(self, _type, properties):
        self.visual_properties[_type] = properties

    def remove_properties(self, _type):
        if _type in self.visual_properties:
            del self.visual_properties[_type]

    def _parse_visual_items(self, graph_item,  _type):
        graph_item.update(self.visual_properties[_type])

    def _parse_visual(self, graph, view):
        for node,data in graph.nodes_iter(data=True):
            if 'label' in data:
                self._parse_visual_items(data, data["label"])
            else:
                print('Warning! Unknown node! (' + str(node) + ')')

        for src,dst,data in graph.edges_iter(data=True):
            if 'label' in data:
                self._parse_visual_items(data, data["label"])
            else:
                print('Warning! Unknown edge! (' + str(src) + "-" + str(dst) + ')')
        return graph

    def print_graph(self, graph, view, trail_name=None, post_name=None, positions=None):
        nx.circular_layout(graph)
        # pyg = nx.nx_pydot.graphviz_layout(graph)
        pyg = nx.drawing.nx_agraph.to_agraph(graph)  # http://stackoverflow.com/questions/35279733/what-could-cause-networkx-pygraphviz-to-work-fine-alone-but-not-together
        pyg.graph_attr.update(graph.graph)
        pyg.layout()

        if positions is not None:
            for n in positions.nodes():
                if n in pyg.nodes():
                    pyg.get_node(n).attr['pos'] = n.attr['pos']

        pyg.layout()
        out = 'images/'

        if post_name == 'colored_union':
            # out += 'allocations/'
            if trail_name is not None:
                out += trail_name
            else:
                raise Exception('Trail name is needed')

        elif view['picture']['name'] == 'allocation':
            out += 'allocations/'
            if trail_name is not None:
                out += trail_name + '_'
            out += view['picture']['name']
            if post_name is not None:
                out += '_' + post_name

        elif view['picture']['name'] == 'higher':
            out += 'highers/'
            if trail_name is not None:
                out += trail_name + '_'
            out += view['picture']['name']
            if post_name is not None:
                out += '_' + post_name

        elif view['picture']['name'] == 'infrastructure':
            if trail_name is not None:
                out += trail_name
            else:
                raise Exception('Trail name is needed')
        else:
            if trail_name is not None:
                out += trail_name + '_'
            out += view['picture']['name']
            if post_name is not None:
                out += '_' + post_name

        out += view['picture']['ext']

        if not os.path.exists(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
        # app = Viewer(graph)
        # app.mainloop()
        #
        pyg.draw(out, format='png', prog='neato', args='-n')
        # pos = nx.graphviz_layout(graph, prog='neatotrail_name')
        # nx.draw(graph, pos)
        # plt.savefig('t.png', dpi=75)
        return pyg


    def _color_by_allocation(self, graph):
        for node in graph.nodes_iter():
            if 'alloc_colors' in graph.node[node].keys() and \
                            len(graph.node[node]['alloc_colors']) > 0 and \
                    (graph.node[node]['label'] == 'Port' or
                             graph.node[node]['label'] == 'Net'):
                num = len(graph.node[node]['alloc_colors'])
                if num == 1:
                    graph.node[node]['style'] = 'filled'
                    col = graph.node[node]['alloc_colors'][0]
                else:
                    graph.node[node]['style'] = 'wedged'
                    col = graph.node[node]['alloc_colors'][0]
                    graph.node[node]['alloc_colors'].pop(0)
                    for color in graph.node[node]['alloc_colors']:
                        col += ';' + str(1.0/num) + ':' + color
                graph.node[node]['fillcolor'] = col

        for start, end, key, edge in graph.edges_iter(keys=True, data=True):
            num = len(edge['alloc_colors'])
            if edge['alloc_colors']:
                col =  edge['alloc_colors'][0]
                edge['alloc_colors'].pop(0)
                for i, color in enumerate(edge['alloc_colors']):
                    col += color
                    if i<len(edge['alloc_colors'])-1:
                        col += ';'
                edge['color'] = col

    def set_color_to_graph(self, graph, color):
        for start, end in graph.edges_iter():
            edge = graph.edge[start][end]
            for key in edge.keys():
                if edge[key]['label'] == 'Flow':
                    edge[key]['style'] = 'bold'
                    edge[key]['color'] = color

        for node in graph.nodes_iter():
            if graph.node[node]['label'] == 'Port' or graph.node[node]['label'] == 'Net':
                graph.node[node]['color'] = color

    def _hide_hidden_nodes(self, graph):
        for node in graph.nodes_iter():
            if 'hidden' in graph.node[node].keys() and graph.node[node]['hidden'] == True:
                graph.node[node]['style'] = 'invis'

    def _set_graph_attributes(self, graph):
        graph.graph['bgcolor'] = '#cdcdb2'
        graph.graph['dpi'] = 75
        graph.graph['size'] = 0.5,0.5
        graph.graph['outputorder'] = 'nodesfirst'

    def refactor_labels(self, graph, view):
        _to_refactor = ['Port', 'Net', 'NF']
        _ignored = ['INode2Node']

        for node in graph.node.keys():
            if graph.node[node]['label'] in _to_refactor:
                graph.node[node]['label'] = node.split('-')[-1]

        if view == GraphView().get_view('infrastructure'):
            for start,end in graph.edges_iter():
                for key in graph.edge[start][end]:
                    if 'bandwidth' in graph.edge[start][end][key] and graph.edge[start][end][key]['bandwidth'] is not None and\
                                    'delay' in graph.edge[start][end][key] and graph.edge[start][end][key]['delay'] is not None:
                        graph.edge[start][end][key]['label'] = graph.edge[start][end][key]['bandwidth']+'\n'+graph.edge[start][end][key]['delay']
                    else:
                        graph.edge[start][end][key]['label'] = ''

        else:
            for start,end in graph.edges_iter():
                for key in graph.edge[start][end]:
                    if graph.edge[start][end][key]['label'] not in _ignored:
                        # if 'bandwidth' in graph.edge[start][end][key]:
                        #     graph.edge[start][end][key]['label'] = graph.edge[start][end][key]['bandwidth']
                        # else:
                        # if 'bandwidth' in graph.edge[start][end][key]:
                        #     graph.edge[start][end][key]['label'] = key+'\n'+graph.edge[start][end][key]['bandwidth']+'\n'+graph.edge[start][end][key]['delay']
                        # else:
                        graph.edge[start][end][key]['label'] = key
                        graph.edge[start][end][key]['fontsize'] = 72
                        graph.edge[start][end][key]['labeldistance'] = 5

    def count_fontcolor(self, graph):
        for node in graph.node:
            actual_node = graph.node[node]
            if 'alloc_colors' in actual_node.keys() and \
                    (actual_node['label'] == 'Port' or actual_node['label'] == 'Net'):
                colors = actual_node['alloc_colors']
                if len(colors) > 0:
                    rgb_colors = []
                    for c in colors:
                        c = c.lstrip('#')
                        rgb_colors.append(tuple(int(c[i:i + 6 // 3], 16) for i in range(0, 6, 6 // 3)))
                    avg = []
                    for r in rgb_colors:
                        avg.append(1 - (0.299 * r[0] + 0.587 * r[1] + 0.114 * r[2])/255)
                    avg = sum(avg)/float(len(avg))
                    if avg > 0.5:
                        actual_node['fontcolor'] = '#FFFFFF'
                    else:
                        actual_node['fontcolor'] = '#000000'
                else:
                    actual_node['fontcolor'] = '#FFFFFF'

    def draw_infra(self, infra, trail_name):
        ygv = GraphView()
        graph_full = ygv.apply_view_filter('higher', infra)
        filtered_infra = ygv.apply_view_filter('infrastructure', infra)
        self.draw(graph_full, ygv.get_view('infrastructure'), trail_name=trail_name)

    def recolor_sap(self, graph):
        for node in graph.node:
            if graph.node[node]['label'] == 'port' and graph.node[node]['properties']['port_type'] == 'enni':
                graph.node[node]['color'] = '#FF0000'

    def fix_infra_edge_coloring(self, graph):
        for start,end in graph.edges_iter():
            for key in graph.edge[start][end]:
                graph.edge[start][end][key]['color'] = graph.edge[start][end][key]['color'].replace('#696969', '')
                if graph.edge[start][end][key]['color'] == '':
                    graph.edge[start][end][key]['color'] = '#FF0000'

    def set_Node_into_INode(self, graph):
        for start, end in graph.edges_iter():
            for key in graph.edge[start][end]:
                if graph.edge[start][end][key]['label'] == 'Flow' and \
                        ((graph.node[start]['label']=='Port' and graph.node[end]['label']=='Net') or
                             (graph.node[start]['label']=='Net' and graph.node[end]['label']=='Port')):
                    graph.edge[start][end][key]['len'] = '4'

    def _color_physical_links(self, graph):
        for start, end in graph.edges_iter():
            edge = graph.edge[start][end]
            for key in edge.keys():
                if edge[key]['label'] == 'Link':
                    edge[key]['color'] = '#5C5C5C'

    def draw(self, graph, view, trail_name=None, post_name=None, color=None, recolored=False, debug_mode=False, positions=None):
        self._set_graph_attributes(graph)
        self._parse_visual(graph, view)
        self.count_fontcolor(graph)
        # self._color_physical_links(graph)
        if view == GraphView().get_view('allocation'):
            if color is not None:
                self.set_color_to_graph(graph, color)
            self._hide_hidden_nodes(graph)
        if view == GraphView().get_view('allocation') and recolored:
            self._color_by_allocation(graph)
            pass
        if not debug_mode:
            self.recolor_sap(graph)
            # self.refactor_labels(graph, view)
            # self.fix_infra_edge_coloring(graph)
            # self.set_Node_into_INode(graph)
        return self.print_graph(graph, view, trail_name=trail_name, post_name=post_name, positions=positions)
