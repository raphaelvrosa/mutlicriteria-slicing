import sys
sys.path.append("/home/raphael/Bkp/HOME/PycharmProjects/slices")


import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, f1_score

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn import utils

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


from sklearn.tree import export_graphviz
from subprocess import call

import networkx as nx


import sys
sys.path.append("/home/raphael/PycharmProjects/slices")

from slicing import Topologies
from mca.topsys import TOPSIS


plt.rcParams['axes.labelsize'] = 28
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24

plt.rcParams['legend.fontsize'] = 24
plt.rcParams['legend.handlelength'] = 2


# to make this notebook's output stable across runs
np.random.seed(42)


# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "gfx")
SLICING_PATH = './data/'


class Status:
    def __init__(self):
        self.data = None
        self.topsys = TOPSIS()
        

        # self.weights = {
        #     'Cost': 0.05,
        #     'Hops': 0.05,
        #     'Latency': 0.6,
        #     'Throughput': 0.05,
        #     'Memory': 0.05,
        #     'CPUs': 0.05,
        #     'Betweeness': 0.05,
        #     'Vitality': 0.05,
        #     'Reachability': 0.05,
        # }

        # self.weights = {
        #     'Cost': 0.05,
        #     'Hops': 0.05,
        #     'Latency': 0.05,
        #     'Throughput': 0.6,
        #     'Memory': 0.05,
        #     'CPUs': 0.05,
        #     'Betweeness': 0.05,
        #     'Vitality': 0.05,
        #     'Reachability': 0.05,
        # }

        self.weights = {
            'Cost': 0.05,
            'Hops': 0.05,
            'Latency': 0.325,
            'Throughput': 0.325,
            'Memory': 0.05,
            'CPUs': 0.05,
            'Betweeness': 0.05,
            'Vitality': 0.05,
            'Reachability': 0.05,
        }

        # self.weights = {
        #     'Cost': 0.1,
        #     'Hops': 0.1,
        #     'Latency': 0.15,
        #     'Throughput': 0.15,
        #     'Memory': 0.1,
        #     'CPUs': 0.1,
        #     'Betweeness': 0.1,
        #     'Vitality': 0.1,
        #     'Reachability': 0.1,
        # }


        self.costs = {
            'Cost': 1,
            'Hops': 1,
            'Latency': 1,
            'Throughput': 0,
            'Memory': 1,
            'CPUs': 1,
            'Betweeness': 0,
            'Vitality': 0,
            'Reachability': 0,
        }

    def rank(self, parsed_data):

        data = parsed_data.values
        cols = [e for e in parsed_data.columns]

        print(parsed_data.columns)
        print('cols', cols)

        self.topsys.set_data(data)

        weights_values = []
        for k in cols:
            weights_values.append(self.weights[k])

        costs_values = []
        for k in cols:
            costs_values.append(self.costs[k])

        print('costs_values', costs_values)

        self.topsys.set_costs(costs_values)
        self.topsys.set_weights(weights_values)

        self.topsys.normalizeMatrix()
        self.topsys.introWeights()
        self.topsys.getIdealSolutions()
        self.topsys.distanceToIdeal()
        
        ranking = self.topsys.relativeCloseness()
 
        #TODO checar ranking NANs e concat se ignore_index produz NANs, entao ver index de parsed_data

        # print 'index parsed', parsed_data.index
        df_score = pd.DataFrame(data=ranking, index=parsed_data.index, columns=['Score'])
        # print df_score.describe()
        # print 'any score nulls', df_score.isnull().values.any()

        rank_df = pd.concat([parsed_data, df_score], axis=1)
        # rank_df = parsed_data.add(df_score)
        return ranking, rank_df

    def finish(self):
        plt.cla()
        plt.clf()
        plt.close()

    def save_fig(self, fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)
        self.finish()

    def load_data(self, filename='full_data.csv', slicing_path=SLICING_PATH):
        csv_path = os.path.join(slicing_path, filename)
        return pd.read_csv(csv_path)

    def info_data(self, data):
        print(data.head())
        print(data.describe()) 
        # print data.info()
        # print data["infra_profile"].value_counts()

    def scatter_data(self, data, feat_x, feat_y, feat_col, feat_hue):
        data.plot(kind="scatter", x=feat_x, y=feat_y, alpha=0.4,
                    s=data[feat_hue], label=feat_hue, figsize=(10,7),
                    c=feat_col, cmap=plt.get_cmap("jet"), colorbar=True,
                    sharex=False)
        plt.legend()
        self.save_fig("scatter_data")

    def hists_data(self, data):
        data.hist(bins=50, figsize=(20,15))
        self.save_fig("histograms_data_plots")

    def scatter_matrix_data(self, data):
        scatter_matrix(data, figsize=(12, 8))
        self.save_fig("scatter_matrix_plot")

    def filter_data(self, data, query):
        #e.g., query = "age > 30 and pets == 0"
        filter_data = data.query(query)
        return filter_data

    def parse_data_features(self, data, keep_features):
        del_cols = [e for e in data.columns if e not in keep_features]
        parsed_data = data.drop(del_cols, axis=1)
        return parsed_data

    def label_dataset(self, data, y_labels):
        del_x_cols = [e for e in data.columns if e not in y_labels]
        y = data.drop(del_x_cols, axis=1)
        x = data.drop(y_labels, axis=1)

        # lab_enc = preprocessing.LabelEncoder()
        lab_enc = preprocessing.KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='uniform')
        y_encoded = lab_enc.fit_transform(y.values)

        print('bin_edges_', lab_enc.bin_edges_)
        print('y_encoded\n', pd.Series(np.ravel(y_encoded)).value_counts())

        label_dataset = {
            'x': x.values,
            'y': np.ravel(y_encoded),
            'features': x.columns, 
            'labels': y.columns,
            # 'classes': [str(c) for c in list(lab_enc.bin_edges_)],
            # 'classes': ['A', 'B', 'C'],
            'classes': ['A', 'B', 'C', 'D', 'E', 'F'],
        }
        return label_dataset

    def split_data(self, label_dataset, test_size=0.2):
        x = label_dataset.get('x')
        y = label_dataset.get('y')
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        # split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

        split_dataset = {
            'features': label_dataset.get('features'), 
            'labels': label_dataset.get('labels'),
            'classes': label_dataset.get('classes'),
            'x_train': X_train, 
            'x_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        }
        return split_dataset

    def forest_data(self, split_dataset):
        rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
        rnd_clf.fit(split_dataset.get('x_train'), split_dataset.get('y_train'))
        y_pred_rf = rnd_clf.predict(split_dataset.get('x_test'))
      
        cfy_features_import = {}
        for name, score in zip(split_dataset.get('features'), rnd_clf.feature_importances_):
            cfy_features_import[name] = score

        cfy_acc = accuracy_score(split_dataset.get('y_test'), y_pred_rf)
        cfy_report_ = metrics.classification_report(split_dataset.get('y_test'), y_pred_rf) 
        
        cfy_report = {
            'features_importance': cfy_features_import,
            'y_pred': y_pred_rf,
            'accuracy': cfy_acc,
            'report': cfy_report_,
            'model': rnd_clf,
        }
        return cfy_report
    

    def pca_explained(self, label_dataset):
        X = label_dataset.get('x')
        #ANALISE PCA
        pca = PCA()
        pca.fit(X)
        #PCA(copy=True, n_components=None, whiten=False)

        plt.bar(np.arange(len(pca.explained_variance_)) + .5, pca.explained_variance_ratio_.cumsum())
        plt.ylim((0, 1))
        plt.xlabel('No. of principal components')
        plt.ylabel('Cumulative variance explained')
        plt.grid(axis = 'y', ls = '-', lw = 1, color = 'white')   
        plt.show()

    def compare_classifiers(self, split_dataset):
        X_train = split_dataset.get("x_train")
        X_test = split_dataset.get("x_test")
        y_train = split_dataset.get("y_train")
        y_test = split_dataset.get("y_test")

        lr = LogisticRegression(solver='lbfgs', multi_class="multinomial")
        # gnb = GaussianNB()
        # svc = LinearSVC(C=1.0, multi_class="crammer_singer")
        rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, max_depth=15, min_samples_leaf=4, random_state=42)
        # rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)

        # best params {'n_estimators': 500, 'max_depth': 15, 'min_samples_leaf': 4}

        # knc = KNeighborsClassifier()
        dtc = DecisionTreeClassifier()

        for clf, name in [(dtc, "Decision Tree"),
                        # (lr, 'Logistic'),
                        # (gnb, 'Naive Bayes'),
                        # (svc, 'Support Vector Classification'),
                        # (knc, 'KNNeighbors'),
                        # (dtc, 'Decision Tree'),
                        (rfc, "RandomForest"),]:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)


            print("%s:" % name)
            print("Features Importance")
            for featname, score in zip(split_dataset.get('features'), clf.feature_importances_):
                print(featname, '-', score)
            print('\n')
            print("\tReport: \n%s" % metrics.classification_report(y_test, y_pred))
            # print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
            # print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
            print("\tAccuracy: %1.3f\n" % accuracy_score(y_test, y_pred))
            if name == "RandomForest":
                print('saving figs')
                self.visual_forest_model(clf, split_dataset, 'random-forest')
                self.confusion_matrix_model(y_test, y_pred)

    def random_forest_grid(self, split_dataset):
        from sklearn.model_selection import GridSearchCV

        X = split_dataset.get("x_train")
        y = split_dataset.get("y_train")
        gridsearch_forest = RandomForestClassifier()

        params = {
            "n_estimators": [100, 300, 500],
            "max_depth": [5,8,15],
            "min_samples_leaf" : [1, 2, 4],
        }
        # print 'grid search'
        clf = GridSearchCV(gridsearch_forest, param_grid=params, cv=5, n_jobs=-1 )
        clf.fit(X,y)
        # print 'best params', clf.best_params_
        # print 'best score', clf.best_score_
        # print 'results', clf.cv_results_

    def confusion_matrix_model(self, y_test, y_pred):
        mat = confusion_matrix(y_test, y_pred)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        self.save_fig("confusion_matrix-random-forest")

    def visual_forest_model(self, model, split_dataset, filename):
        feature_names = list(split_dataset.get("features"))
        target_names = split_dataset.get("classes")
        estimator = model.estimators_[5]

        # Export as dot file
        export_graphviz(estimator, out_file='./gfx/tree.dot', 
                        feature_names = feature_names,
                        class_names = target_names,
                        rounded = True, proportion = False, 
                        precision = 2, filled = True)

        # Convert to png using system command (requires Graphviz)
        # call(['dot', '-Tpng', './gfx/tree.dot', '-o', './gfx/tree.png', '-Gdpi=600'])
        call(['dot', '-Tpdf', './gfx/tree.dot', '-o', './gfx/tree-'+filename+'.pdf', '-Gdpi=300'])

    def visual_forest_data(self, split_dataset, cfy_report):
        model = cfy_report.get("model")
        feature_names = list(split_dataset.get("features"))
        target_names = split_dataset.get("classes")
        estimator = model.estimators_[5]

        # Export as dot file
        export_graphviz(estimator, out_file='./gfx/tree.dot', 
                        feature_names = feature_names,
                        class_names = target_names,
                        rounded = True, proportion = False, 
                        precision = 2, filled = True)

        # Convert to png using system command (requires Graphviz)
        # call(['dot', '-Tpng', './gfx/tree.dot', '-o', './gfx/tree.png', '-Gdpi=600'])
        call(['dot', '-Tpdf', './gfx/tree.dot', '-o', './gfx/tree.pdf', '-Gdpi=600'])

    def confusion_matrix(self, split_dataset, cfy_report):
        mat = confusion_matrix(split_dataset.get('y_test'), cfy_report.get('y_pred'))
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        self.save_fig("confusion_matrix")

    def pca_analysis(self, label_dataset):
        pca = PCA(2)  # project to 2 dimensions
        projected = pca.fit_transform(label_dataset.get('x'))       
        # plt.scatter(projected[:, 0], projected[:, 1],
        #             c=label_dataset.get('y'), edgecolor='none', alpha=0.5,
        #             cmap=plt.cm.get_cmap('spectral', 10))
        # plt.xlabel('component 1')
        # plt.ylabel('component 2')
        # plt.colorbar()
        # self.save_fig('pca_data')

        pca_analysis = {
            'init_shape': label_dataset.get('x').shape,
            'projected_shape': projected.shape,
            'components': pca.components_,
            'explained_variance': pca.explained_variance_ratio_,
        }
        return pca_analysis

    def tsne_analysis(self, label_dataset):
        tsne = TSNE(n_components=2, random_state=42)
        X_reduced = tsne.fit_transform(label_dataset.get('x'))
        plt.figure(figsize=(13,10))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=label_dataset.get('y'), cmap="jet")
        plt.axis('off')
        plt.colorbar()
        self.save_fig('tsne_data')

    def boxplot(self, data):
        filename = "boxplot"
        # ax = sns.heatmap(self.data, cmap='RdYlGn_r')
        plt.figure(figsize=(10, 6))
        df_norm = (data - data.mean()) / (data.max() - data.min())
        ax = sns.boxplot(data=df_norm)

        # turn the axis label
        # for item in ax.get_yticklabels():
        #     item.set_rotation(0)
        #
        # ax = sns.violinplot(data=df_norm, inner=None)
        # ax = sns.swarmplot(data=df_norm,
        #               alpha=0.7,
        #               color="white", edgecolor="gray", cmap="jet")
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        self.save_fig(filename)

    def factorplot(self, data, x, y, col, hue, prefix):
        filename = 'factor_plot' + '_' + prefix

        sns.factorplot(x=x,
                       y=y,
                       data=data,
                       hue=hue,  # Color by stage
                       col=col,  # Separate by stage
                       kind='bar',
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

    def jointplot(self, feat1, feat2, data, prefix):
        filename = 'jointplot' + '_' + prefix
        sns.jointplot(x=feat1, y=feat2, data=data, kind="reg",
                      marginal_kws = dict(bins=15, rug=True), annot_kws = dict(stat="r"))
        self.save_fig(filename)

    def heatmap(self, data):
        filename = "heatmap"
        # ax = sns.heatmap(self.data, cmap='RdYlGn_r')
        df_norm = (data - data.mean()) / (data.max() - data.min())
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
        self.save_fig(filename)

    def scatter(self, data, x, y, s, c, prefix):
        filename = "scatterplot" + '_' + prefix
        data.plot(kind="scatter", x=x, y=y, alpha=0.4,
              s=data[s], label=s, figsize=(10, 7), # data[s]*7 if mining_102
              c=c, cmap=plt.get_cmap("jet"), colorbar=True,
              sharex=False)
        plt.legend()
        # plt.ylim((0.0,0.0005))
        self.save_fig(filename)

    def mining_101(self, data, query):
        filtered_data = self.filter_data(data, query)
        self.factorplot(filtered_data, 'Infra_Policy', 'Packet_Loss', 'NFs', 'Infra_Profile', 'flr_01')
        self.factorplot(filtered_data, 'Infra_Policy', 'Throughput', 'NFs', 'Infra_Profile', 'through_01')
        self.factorplot(filtered_data, 'Infra_Policy', 'Latency', 'NFs', 'Infra_Profile', 'latency_01')
        self.factorplot(filtered_data, 'Infra_Policy', 'Cost', 'NFs', 'Infra_Profile', 'cost_01')

    def mining_102(self, data, query):
        filtered_data = self.filter_data(data, query)
        self.info_data(filtered_data)
        
        self.scatter(filtered_data, 'Throughput', 'Latency', 'CPUs', 'Cost', 'flr_01')
        self.scatter(filtered_data, 'Hops', 'Latency', 'CPUs', 'Cost', 'flr_02')
        # self.scatter(filtered_data, 'CPUs', 'Latency', 'Latency', 'Cost', 'flr_05')
        # self.scatter(filtered_data, 'Memory', 'Throughput', 'Latency', 'Cost', 'flr_04')
        # self.scatter(filtered_data, 'Memory', 'vitality', 'Latency', 'Cost', 'flr_06')
        # self.scatter(filtered_data, 'CPUs', 'betweeness', 'Latency', 'Cost', 'flr_07')
        # self.scatter(filtered_data, 'Latency', 'betweeness', 'Hops', 'Cost', 'flr_08')
        # self.scatter(filtered_data, 'Latency', 'vitality', 'Hops', 'Cost', 'flr_09')
        self.scatter(filtered_data, 'Betweeness', 'Latency', 'CPUs', 'Cost', 'flr_10')
        self.scatter(filtered_data, 'Vitality', 'Latency', 'CPUs', 'Cost', 'flr_11')
        self.scatter(filtered_data, 'Betweeness', 'Throughput', 'CPUs', 'Cost', 'flr_12')
        self.scatter(filtered_data, 'Vitality', 'Throughput', 'CPUs', 'Cost', 'flr_13')

        self.scatter(filtered_data, 'Betweeness', 'Hops', 'CPUs', 'Cost', 'flr_16')
        self.scatter(filtered_data, 'Vitality', 'Hops', 'CPUs', 'Cost', 'flr_17')

        # self.scatter(filtered_data, 'betweeness', 'Cost', 'CPUs', 'Latency', 'flr_16')
        # self.scatter(filtered_data, 'vitality', 'Cost', 'CPUs', 'Throughput', 'flr_17')


        # self.jointplot('CPUs', 'Cost', data, 'cpus_vs_cost')
        # self.jointplot('Memory', 'Latency', data, 'mem_vs_latency')

    def mining_103(self, data, query):
        filtered_data = self.filter_data(data, query)
        self.factorplot(filtered_data, 'Infra_Policy', 'reachability', 'NFs', 'Infra_Profile', 'reachability')
        self.factorplot(filtered_data, 'Infra_Policy', 'vitality', 'NFs', 'Infra_Profile', 'vitality')
        self.factorplot(filtered_data, 'Infra_Policy', 'betweeness', 'NFs', 'Infra_Profile', 'betweeness')
        self.factorplot(filtered_data, 'Infra_Policy', 'density', 'NFs', 'Infra_Profile', 'density')
        self.factorplot(filtered_data, 'Infra_Policy', 'availability', 'NFs', 'Infra_Profile', 'availability')

    def mining_104(self, data, query):
        filtered_data = self.filter_data(data, query)
        self.factorplot(filtered_data, 'Infra_Profile', 'Packet_Loss', 'Infra_Policy', 'Infra_Model', 'flr_0104')
        self.factorplot(filtered_data, 'Infra_Profile', 'Throughput', 'Infra_Policy', 'Infra_Model', 'through_0104')
        self.factorplot(filtered_data, 'Infra_Profile', 'Latency', 'Infra_Policy', 'Infra_Model', 'latency_0104')
        self.factorplot(filtered_data, 'Infra_Profile', 'Cost', 'Infra_Policy', 'Infra_Model', 'cost_0104')


class GraphLookup:
    def __init__(self):
        self.topos = Topologies()
        self.graph = None

    def load_graph(self, filename):
        self.graph = self.topos.load(filename, base=True)

    def measures(self, measure):
        mes = None
        if measure == 'neigh_degree':
            mes = nx.average_neighbor_degree(self.graph)
        elif measure == 'vitality':
            mes = nx.closeness_vitality(self.graph)
        elif measure == 'centrality':
            mes = nx.closeness_centrality(self.graph)
        elif measure == 'betweeness':
            mes = nx.betweenness_centrality(self.graph)
        elif measure == 'degree':
            mes = nx.degree(self.graph)
        return mes

    def prep_mes(self, mes):
        sequence=sorted(mes.values(),reverse=True) # degree sequence
        dmax=max(sequence)
        # print dmax
        return sequence, dmax

    def dist_between(self, filename):
        self.load_graph(filename)
        mes = self.measures('betweeness')
        sequence, dmax = self.prep_mes(mes)
        plt.hist(sequence,bins=30)
        plt.title("Betweeness histogram")
        plt.ylabel("count")
        plt.xlabel("betweeness")
        self.draw_graph()

    def dist_degree(self, filename):
        self.load_graph(filename)
        mes = self.measures('degree')
        sequence, dmax = self.prep_mes(mes)
        plt.hist(sequence,bins=dmax)
        plt.title("Degree histogram")
        plt.ylabel("count")
        plt.xlabel("degree")
        self.draw_graph()

    def draw_graph(self):
        plt.axes([0.45,0.45,0.45,0.45])
        Gcc=nx.connected_component_subgraphs(self.graph)[0]
        pos=nx.spring_layout(Gcc)
        plt.axis('off')
        nx.draw_networkx_nodes(Gcc,pos,node_size=20)
        nx.draw_networkx_edges(Gcc,pos,alpha=0.4)
        # plt.savefig("degree_histogram.png")
        plt.show()

    def dist_vitality(self, filename):
        self.load_graph(filename)
        mes = self.measures('vitality')
        sequence, dmax = self.prep_mes(mes)
        plt.hist(sequence,bins=30)
        plt.title("Vitality histogram")
        plt.ylabel("count")
        plt.xlabel("vitality")
        self.draw_graph()


    def dist_centrality(self, filename):
        self.load_graph(filename)
        mes = self.measures('centrality')
        sequence, dmax = self.prep_mes(mes)
        plt.hist(sequence,bins=30)
        plt.title("centrality histogram")
        plt.ylabel("count")
        plt.xlabel("centrality")
        self.draw_graph()

    def lookup(self):
        name = 'topo_100_4_3'
        # self.dist_between(name)
        # self.dist_vitality(name)
        self.dist_centrality(name)


class Lookup:

    # full_features = ,index,availability,cost,density,disk_size,frame_loss_ratio,hops,latency,mem_size,num_cpus,reachability,robustness,throughput,vitality,paths,model,size,infra_nodes_rate,infra_mode,infra_profile,service_nfs,exp_id
    
    def __init__(self):
        self.sts = Status()

    def simple_pipe(self):
        
        # query = "Infra_Model == 1 and size == 100"
        query = 'Infra_Model == 1 and size == 100 and Infra_Policy == "centrality" and NFs == 3'
        
        # query = 'Infra_Model==4 and Infra_Profile == 3 and Infra_Policy == "centrality" and infra_nodes_rate == 0.1 and size == 100 and NFs == 3'

        # query = 'Infra_Profile == 3 and Infra_Policy == "centrality" and infra_nodes_rate == 0.1 and size == 100'

        # features = ['Cost','Hops','Latency','Memory','CPUs','Throughput']


        # MCA query
        features = ['Cost','Hops','Latency','Throughput','Memory','CPUs', 'Betweeness', 'Vitality', 'Reachability']

        # ML query
        # features = ['Infra_Profile', 'Cost','Hops','Latency','Throughput','Memory','CPUs', 'Betweeness', 'Vitality', 'Reachability']

        # features = ['Cost','Hops','Latency','Throughput','Memory','CPUs']
        y_labels = ['Latency']

        data = self.sts.load_data()
        
        data.rename(columns={
            'infra_mode': 'Infra_Policy',
            'infra_profile': 'Infra_Profile',
            'latency':'Latency',
            'throughput':'Throughput',
            'frame_loss_ratio': 'Packet_Loss',
            'service_nfs': 'NFs',
            'model': 'Infra_Model',
            'hops': 'Hops',
            'cost': 'Cost',
            'mem_size': 'Memory',
            'num_cpus': 'CPUs',
            'robustness': 'Betweeness',
            'vitality': 'Vitality', 
            'reachability':'Reachability',
        }, inplace=True)


        # self.sts.mining_101(data, 'Infra_Model==1 and infra_nodes_rate == 0.1 and size == 100')
        # self.sts.mining_102(data, 'Infra_Model==4 and Infra_Profile == 3 and Infra_Policy == "centrality" and infra_nodes_rate == 0.1 and size == 100 and NFs == 3')
        # self.sts.mining_103(data, 'infra_nodes_rate == 0.1 and size == 100')
        # self.sts.mining_104(data, 'NFs == 3 and infra_nodes_rate == 0.1 and size == 100')

        filtered_data = self.sts.filter_data(data, query)
        parsed_data = self.sts.parse_data_features(filtered_data, features)
        # self.sts.info_data(parsed_data)
        # print parsed_data.index

        # print 'any nulls', parsed_data.isnull().values.any()
        # print 'any vitality nulls', parsed_data['Vitality'].isnull().values.any()
        
        ranking, rank_df = self.sts.rank(parsed_data)
        # self.sts.info_data(rank_df)
        # print 'any rank_df nulls', rank_df.isnull().values.any()
        # print 'any rank_df score nulls', rank_df['Score'].isnull().values.any()
        # print 'ranking\n', ranking
        # self.sts.boxplot(rank_df)
        # self.sts.hists_data(rank_df)
        # self.sts.scatter_matrix_data(rank_df)
        # self.sts.heatmap(rank_df)
        # self.sts.scatter(rank_df, 'Hops', 'Throughput', 'Cost', 'Score', 'score_01-t')

        # self.sts.scatter(rank_df, 'Throughput', 'Latency', 'Cost', 'Score', 'score_02-t')
        # self.sts.scatter(rank_df, 'Vitality', 'Throughput', 'Cost', 'Score', 'score_04-t')

        # self.sts.scatter(rank_df, 'Throughput', 'Latency', 'Cost', 'Score', 'score_02-l')
        # self.sts.scatter(rank_df, 'Vitality', 'Latency', 'Cost', 'Score', 'score_04-l')

        self.sts.scatter(rank_df, 'Throughput', 'Latency', 'Cost', 'Score', 'score_02-e')
        self.sts.scatter(rank_df, 'Hops', 'Latency', 'Throughput', 'Score', 'score_04-e')


        # self.sts.info_data(parsed_data)
        # self.sts.boxplot(parsed_data)
        # self.sts.scatter_data(parsed_data, 'num_cpus', 'latency', 'cost', 'throughput')
        # self.sts.hists_data(parsed_data)
        # self.sts.scatter_matrix_data(parsed_data)

        # labeled_data = self.sts.label_dataset(parsed_data, y_labels)
        # split_data = self.sts.split_data(labeled_data)
        # self.sts.compare_classifiers(split_data)

        # self.sts.random_forest_grid(split_data)

        # cfy_report = self.sts.forest_data(split_data)
        # print cfy_report
        # self.sts.visual_forest_data(split_data, cfy_report)
        # self.sts.confusion_matrix(split_data, cfy_report)

        # pca_analysis = self.sts.pca_analysis(labeled_data)
        # print pca_analysis
        # self.sts.tsne_analysis(labeled_data)





if __name__ == '__main__':
    lkp = Lookup()
    lkp.simple_pipe()

    # glkp = GraphLookup()
    # glkp.lookup()