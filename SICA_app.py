import pandas as pd 
from sklearn.decomposition import PCA
import numpy as np 
from collections import Counter
from scipy.spatial import distance_matrix
from scipy import optimize
from bokeh.models import (LassoSelectTool, PanTool,
                          ResetTool, PolySelectTool,
                          HoverTool, WheelZoomTool, ColumnDataSource,
                          TapTool, Circle, Text, Div, NumberFormatter)
TOOLS = [LassoSelectTool, PanTool, WheelZoomTool, ResetTool]
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot, widgetbox, layout, column, row
from bokeh.palettes import Category20
from bokeh.models.callbacks import CustomJS, OpenURL
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn, Slider, Button, Div, Dropdown, Select
from bokeh.io import curdoc
import random
from scipy.spatial.distance import cdist, squareform
import matlab.engine 

class Table():
    def __init__(self, name, pca, sica):
        self.name = name
        self.pca_weights = pca
        self.sica_weights = sica

    def get_dict(self):
        data = {
            'name': self.name, #+ ["chroma_mean{}".format(x) for x in range(0, 12)] + ["chroma_std{}".format(x) for x in range(0, 12)],
            'pca_weightcomp1': list(self.pca_weights[0, :]),
            'pca_weightcomp2': list(self.pca_weights[1, :]),
            'sica_weightcomp1': list(self.sica_weights[0, :]),
            'sica_weightcomp2': list(self.sica_weights[1, :])
        }
        return data


class SICA():
    def __init__(self):
        self.X = 0
        self.L = 0
        self.l1 = 0
        self.l2 = 0
        self.eig = []
        self.num_edges = 0
        self.n = 0
        self.I = 0
        self.d = 0
        self.W = 0
        self.eng = matlab.engine.start_matlab()

    def fit_transform(self, X, L):
        self.X = X.T.astype(np.float64)
        self.L = L
        self.d = X.shape[1]
        self.n = X.shape[0]
        self.num_edges = np.trace(L)/2
        self.I = np.identity(self.n)
        eig, vec = np.linalg.eig(L)
        self.eig = np.absolute(np.real(eig).flatten())
        #res = optimize.fmin_powell(self.lagrange, [0.1, 0.1])
        #res = optimize.fmin_bfgs(self.lagrange, [0.1, 0.1], fprime=self.grad)
        #res = optimize.minimize(self.lagrange, [0.1, 0.1], method='trust-ncg',
        #                        jac=self.grad, hess=self.hess,
        #                        options={'xtol': 1e-8, 'disp': True, 'maxiter':1000})
        W, res = self.eng.findOptimumW(matlab.double(self.X.T.tolist()), matlab.double(self.L.tolist()), nargout=2) 
        res = np.array(res._data)
        self.l1, self.l2 = res[0], res[1]
        W = np.array(W._data).reshape(W.size, order='F')
        self.components_ = W
        return np.matmul(self.X.T, W).view(type=np.ndarray)

    def update_l_transform(self, l1, l2):
        if l1 != self.l1 or l2 != self.l2:
            self.l1, self.l2 = l1, l2
            W, res= self.eng.findOptimumW(matlab.double(self.X.T.tolist()), matlab.double(self.L.tolist()), matlab.double([l1, l2]), nargout=2)
            W = np.array(W._data).reshape(W.size, order='F')
            self.components_ = W
        return np.matmul(self.X.T, W).view(type=np.ndarray)
'''
def create_laplacian(valence):
    L = np.zeros((len(valence), len(valence)), dtype=int)
    mean = np.mean(valence)
    groups = np.zeros((len(valence)))
    groups[valence>mean] = 1
    count = Counter(groups)
    for key, value in count.items():
        ind = [i for i in range(0, len(valence)) if groups[i] == key]
        for i in ind:
            L[i, ind] = 1
    return L
'''
def create_laplacian(feat):
    dist = np.zeros((len(feat), len(feat)))
    for i in range(0, len(feat)):
        distvec = list(np.absolute(np.array(feat)-feat[i]))
        for k in range(0, len(distvec)):
            dist[i, k] = distvec[k]
    e = 0.1
    ind = dist<e 
    ind2 = dist>=e 
    dist[ind] = 1
    dist[ind2] = 0
    np.fill_diagonal(dist, 0)
    return dist
# List of features to use in PCA and SICA from the dataframe
spot_features = ['danceability', 'energy', 'valence', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo']
features = [x for x in spot_features if x!= 'energy']

genres = ['Folk', 'Electronic']

df = pd.read_hdf('/Users/ah13558/Documents/PhD/interesting-svd/data/SICA_songs.h5', 'df') # Load in dataframe which contains youtube & features 
df = df.loc[df['genre'].isin(genres)]
df = df.groupby('genre').head(100).reset_index(drop=True) # Take 50 of each genre
df['color'] = [Category20[20][x*2] for x in list(df.genre.astype('category').cat.codes)]
df['youtube_embed'] = df.youtube.apply(lambda x: (x.replace("watch?v=", "embed/") + "?autoplay=1"))

#Standardise df features
scaler = StandardScaler()
df[spot_features] = scaler.fit_transform(df[spot_features])
spot_features += ['key_{}'.format(i) for i in range(0, 12)]
spot_features += ['mode']

features += ['key_{}'.format(i) for i in range(0, 12)]
features += ['mode']
# One-hot embed the key
df = pd.concat([df,pd.get_dummies(df['key'], prefix='key')],axis=1)

# Include the mean and standard deviation of each 12 bins in chromagram for whole song
chromas = list(df.chromas) 
chromas_m = np.ndarray((len(chromas), len(chromas[0][0])*2))
for i in range(0, len(chromas)):
    chromas_m[i, :] = np.concatenate((chromas[i][0], chromas[i][1]))
#chromas_m = scaler.fit_transform(chromas_m)

# Create data matrix with chromagram features and spotify features
X = np.hstack((df[features],chromas_m))
X = df[features].values
# Perform PCA and plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df['x_pca'] = X_pca[:, 0]
df['y_pca'] = X_pca[:, 1]

sica = SICA()
lap = create_laplacian(list(df.energy))
X_sica = sica.fit_transform(X, lap)
df['x_sica'] = X_sica[:, 0]
df['y_sica'] = X_sica[:, 1]
score1, score2 = [], []
for i in range(0,10):
    X_train, X_test, Y_train, Y_test = train_test_split(df[['x_pca', 'y_pca']], df.genre.astype('category').cat.codes)
    log1 = LogisticRegression()
    log1.fit(X_train, Y_train)
    score1.append(log1.score(X_test, Y_test))

    X_train, X_test, Y_train, Y_test = train_test_split(df[['x_sica', 'y_sica']], df.genre.astype('category').cat.codes)
    log2 = LogisticRegression()
    log2.fit(X_train, Y_train)
    score2.append(log2.score(X_test, Y_test))
s1, s2 = np.mean(score1), np.mean(score2)
std1, std2 = np.std(score1), np.std(score2)
# PLOTTING
tooltip = """
        <div>
            <div>
               <iframe width="420" height="192" src=@youtube_embed frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
            </div>
            <div>
                <span style="font-size: 17px;">@artist_name, @song_title</span>
            </div>
        </div>
           """
source = ColumnDataSource(df)

p1 = figure(plot_width=600, plot_height=500, title='PCA',
         toolbar_location="below", tools="tap")
p2 = figure(plot_width=600, plot_height=500, title='SICA',
        toolbar_location="below", tools="tap")

c1 = p1.circle('x_pca', 'y_pca', size=10, color='color',
            legend='genre', source=source, hover_alpha=0.5, hover_fill_color='color', hover_line_color='color')
c2 = p2.circle('x_sica', 'y_sica', size=10, color='color',
            legend='genre', source=source, hover_alpha=0.5, hover_fill_color='color', hover_line_color='color')
c3 = p1.circle('x_pca', 'y_pca', size=30, alpha=0, color='color',
            legend='genre', source=source, hover_alpha=0.5, hover_fill_color='color', hover_line_color='color')
c4 = p2.circle('x_sica', 'y_sica', size=30, alpha=0, color='color',
            legend='genre', source=source, hover_alpha=0.5, hover_fill_color='color', hover_line_color='color')
nonselect = Circle(fill_alpha=0, line_alpha=0)
c3.nonselection_glyph = nonselect
c4.nonselection_glyph = nonselect
code1 = "source1.set('selected', cb_data['index']);"
code2 = "source2.set('selected', cb_data['index']);"
callback = CustomJS(args={'source1': source, 'source2': source}, code=code1+code2)

hover1 = HoverTool(callback=callback, renderers=[c1], tooltips=tooltip)
hover2 = HoverTool(callback=callback, renderers=[c2], tooltips=tooltip)
taptool = p1.select(type=TapTool)
taptool.callback = OpenURL(url="@youtube")
taptool = p2.select(type=TapTool)
taptool.callback = OpenURL(url="@youtube")
p1.add_tools(hover1)
p2.add_tools(hover2)

table = Table(features, pca.components_, sica.components_[0:2, :])
sourcetab = ColumnDataSource(table.get_dict())
format = NumberFormatter(format='0.00000')
columns = [TableColumn(field='name', title='Feature Name'),
           TableColumn(field='pca_weightcomp1', title='PCA x-axis Weight', formatter=format),
           TableColumn(field='pca_weightcomp2', title='PCA y-axis Weight', formatter=format),
           TableColumn(field='sica_weightcomp1', title='SICA x-axis Weight', formatter=format),
           TableColumn(field='sica_weightcomp2', title='SICA y-axis Weight', formatter=format)]

data_table = DataTable(source=sourcetab, columns=columns, width=700, height=600)
'''
g1 = figure(plot_width=600, plot_height=500, x_range=(-1.1,1.1), y_range=(-1.1,1.1))
g_render = from_networkx(G, nx.spring_layout, scale=2, center=(0,0))
g1.renderers.append(g_render)
'''
# BOKEH SERVER
lambda1 = Slider(title="lambda1", value=sica.l1, start=0, end=1, step=0.01)
lambda2 = Slider(title="lambda2", value=sica.l2, start=0, end=1, step=0.01)

button = Button(label="Recalculate", button_type='primary', width=100)

div = Div(text="""Logistic regression accuracy using PCA: %.2f+-%.2f<br /> Linear classifier accuracy using SICA: %.2f+-%.2f"""% (s1, std1, s2, std2),
    width=400, height=100)
menu = [(x.title(), x) for x in spot_features]
dropdown = Select(title="Select Regulariser", value='energy', options=spot_features, width=300)

def update_data():
    X_sica = sica.update_l_transform(lambda1.value, lambda2.value)
    df['x_sica'] = X_sica[:, 0]
    df['y_sica'] = X_sica[:, 1]
    source.data = ColumnDataSource.from_df(df)
    table.sica_weights = sica.components_[0:2, :]
    sourcetab.data = table.get_dict()
    score2 = []
    for i in range(0,10):
        X_train, X_test, Y_train, Y_test = train_test_split(df[['x_sica', 'y_sica']], df.genre.astype('category').cat.codes)
        log2 = LogisticRegression()
        log2.fit(X_train, Y_train)
        score2.append(log2.score(X_test,Y_test))
    s2 = np.mean(score2)
    std2 = np.std(score2)
    div.text = """Logistic regression accuracy using PCA: %.2f+-%.2f<br /> Linear classifier accuracy using SICA: %.2f+-%.2f"""% (s1, std1, s2, std2)

def update_regulariser(attr, old, new):
    reg = dropdown.value
    features = [x for x in spot_features if x != reg]
    X = df[features].values
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df['x_pca'] = X_pca[:, 0]
    df['y_pca'] =  X_pca[:, 1]
    X_sica = sica.fit_transform(X, create_laplacian(list(df[reg])))
    df['x_sica'] = X_sica[:, 0]
    df['y_sica'] = X_sica[:, 1]
    source.data = ColumnDataSource.from_df(df)
    table.name = features
    table.pca_weights = pca.components_
    table.sica_weights = sica.components_[0:2, :]
    sourcetab.data = table.get_dict()
    score2 = []
    for i in range(0,10):
        X_train, X_test, Y_train, Y_test = train_test_split(df[['x_pca', 'y_pca']], df.genre.astype('category').cat.codes)
        log1 = LogisticRegression()
        log1.fit(X_train, Y_train)
        score1.append(log1.score(X_test, Y_test))
        X_train, X_test, Y_train, Y_test = train_test_split(df[['x_sica', 'y_sica']], df.genre.astype('category').cat.codes)
        log2 = LogisticRegression()
        log2.fit(X_train, Y_train)
        score2.append(log2.score(X_test,Y_test))
    s1, s2 = np.mean(score1), np.mean(score2)
    std1, std2 = np.std(score1), np.std(score2)
    div.text= """Logistic regression accuracy using PCA: %.2f+-%.2f<br /> Linear classifier accuracy using SICA: %.2f+-%.2f"""% (s1, std1, s2, std2)
    lambda1.value, lambda2.value = sica.l1, sica.l2

dropdown.on_change('value', update_regulariser)
button.on_click(update_data)
l = layout([
    row([p1, p2], sizing_mode="stretch_both"), 
    row(widgetbox(data_table), column(widgetbox(dropdown, lambda1, lambda2, button, div), sizing_mode="scale_width"), sizing_mode="scale_width")
    ], sizing_mode='stretch_both')
curdoc().add_root(l)
curdoc().title = "PCA v SICA for music"

