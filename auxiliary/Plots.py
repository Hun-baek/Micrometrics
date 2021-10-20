import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import geopandas as gpd
from scipy.stats import probplot
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from IPython.core.interactiveshell import InteractiveShell

def fig1(df_fig1) :
    
    sns.set(style = 'whitegrid')
    
    df_fig1['country'] = df_fig1['country'].map(str.upper)
    df_fig1['country'] = df_fig1['country'].str[:3]
    
    fig, ax = plt.subplots(1,3, figsize = (16, 5),
                           constrained_layout = True)
    
    ax[0].set_title('Ethnicity', size = 20)
    ax[0].scatter(df_fig1['ethnicity_C2'],df_fig1['ethnicity_C'], s = 5)
    ax[0].set_ylabel("$\\tilde{S}$", size = 18)
    ax[0].set_xlabel("$\hat{S}$", size = 18)
    for i in range(df_fig1.shape[0]):
        ax[0].annotate(df_fig1.country[i],
                         (df_fig1.ethnicity_C2[i],df_fig1.ethnicity_C[i]),
                         ha = 'left',
                         va = 'center')
    
    ax[1].set_title('Language', size = 20)
    ax[1].scatter(df_fig1['language_C2'],df_fig1['language_C'], s = 5)
    ax[1].set_ylabel("$\\tilde{S}$", size = 18)
    ax[1].set_xlabel("$\hat{S}$", size = 18)
    for i in range(df_fig1.shape[0]):
        ax[1].annotate(df_fig1.country[i],
                         (df_fig1.language_C2[i],df_fig1.language_C[i]),
                         ha = 'left',
                         va = 'center')
    
    ax[2].set_title('Religion',size = 20)
    ax[2].scatter(df_fig1['religion_C2'],df_fig1['religion_C'], s = 5)
    ax[2].set_ylabel("$\\tilde{S}$", size = 18)
    ax[2].set_xlabel("$\hat{S}$", size = 18)
    for i in range(df_fig1.shape[0]):
        ax[2].annotate(df_fig1.country[i],
                         (df_fig1.religion_C2[i],df_fig1.religion_C[i]),
                         ha = 'left',
                         va = 'center')
    
    return plt.show()



def fig2(df_fig2) :
    
    sns.set(style = 'whitegrid')
    
    df_fig2['country'] = df_fig2['country'].map(str.upper)
    df_fig2['country'] = df_fig2['country'].str[:3]
    
    fig, ax = plt.subplots(1,3, figsize = (16, 5),
                           constrained_layout = True)
    
    ax[0].scatter(df_fig2['language_C2'],df_fig2['ethnicity_C2'],s = 5)
    ax[0].set_ylabel("Ethnicity $\hat{S}$", size = 18)
    ax[0].set_xlabel("Language $\hat{S}$", size = 18)
    for i in range(df_fig2.shape[0]):
        ax[0].annotate(df_fig2.country[i],
                         (df_fig2.language_C2[i],df_fig2.ethnicity_C2[i]),
                         ha = 'left',
                         va = 'center')
    
    
    ax[1].scatter(df_fig2['language_C2'],df_fig2['religion_C2'], color = 'blue', s = 5)
    ax[1].set_ylabel("Religion $\hat{S}$", size = 18)
    ax[1].set_xlabel("Language $\hat{S}$", size = 18)
    for i in range(df_fig2.shape[0]):
        ax[1].annotate(df_fig2.country[i],
                         (df_fig2.language_C2[i],df_fig2.religion_C2[i]),
                         ha = 'left',
                         va = 'center')
    ax[2].scatter(df_fig2['ethnicity_C2'],df_fig2['religion_C2'], color = 'blue', s = 5)
    ax[2].set_ylabel("Religion $\hat{S}$", size = 18)
    ax[2].set_xlabel("Ethinicity $\hat{S}$", size = 18)
    for i in range(df_fig2.shape[0]): 
        ax[2].annotate(df_fig2.country[i],
                         (df_fig2.ethnicity_C2[i],df_fig2.religion_C2[i]),
                         ha = 'left',
                         va = 'center')
        
    return plt.show()



def fig3(df_fig3) : 
    
    sns.set(style = 'whitegrid')
    
    df_fig3['country'] = df_fig3['country'].map(str.upper)
    df_fig3['country'] = df_fig3['country'].str[:3]
    
    fig, ax = plt.subplots(3,2, figsize = (12,18))

    sns.regplot(x = "ethnicity_I", y = "ethnicity_C2", data = df_fig3, ax = ax[0][0])
    ax[0][0].set_ylabel("Ethnicity $\hat{S}$", size = 14)
    ax[0][0].set_xlabel("Ethnicity $F$", size = 14)
    for i in range(df_fig3.shape[0]):
        ax[0][0].annotate(df_fig3.country[i],
                         (df_fig3.ethnicity_I[i],df_fig3.ethnicity_C2[i]),
                         ha = 'left',
                         va = 'center')
    
    sns.regplot(x = "lnGDP_pc",  y = "ethnicity_C2", data = df_fig3, ax = ax[0][1])
    ax[0][1].set_ylabel("Ethnicity $\hat{S}$", size = 14)
    ax[0][1].set_xlabel("Log per capita GDP", size = 14)
    for i in range(df_fig3.shape[0]):
        ax[0][1].annotate(df_fig3.country[i],
                         (df_fig3.lnGDP_pc[i],df_fig3.ethnicity_C2[i]),
                         ha = 'left',
                         va = 'center')
        
    sns.regplot(x = "language_I", y = "language_C2", data = df_fig3, ax = ax[1][0])
    ax[1][0].set_ylabel("Language $\hat{S}$", size = 14)
    ax[1][0].set_xlabel("Language $F$", size = 14)
    for i in range(df_fig3.shape[0]):
        ax[1][0].annotate(df_fig3.country[i],
                         (df_fig3.language_I[i],df_fig3.language_C2[i]),
                         ha = 'left',
                         va = 'center')
        
    sns.regplot(x = "lnGDP_pc", y = "language_C2", data = df_fig3,ax = ax[1][1])
    ax[1][1].set_ylabel("Language $\hat{S}$", size = 14)
    ax[1][1].set_xlabel("Log per capita GDP", size = 14)
    for i in range(df_fig3.shape[0]):
        ax[1][1].annotate(df_fig3.country[i],
                         (df_fig3.lnGDP_pc[i],df_fig3.language_C2[i]),
                         ha = 'left',
                         va = 'center')
        
    sns.regplot(x = "religion_I", y = "religion_C2", data = df_fig3, ax = ax[2][0])
    ax[2][0].set_ylabel("Religion $\hat{S}$", size = 14)
    ax[2][0].set_xlabel("Religion $F$", size = 14)
    for i in range(df_fig3.shape[0]):
        ax[2][0].annotate(df_fig3.country[i],
                         (df_fig3.religion_I[i],df_fig3.religion_C2[i]),
                         ha = 'left',
                         va = 'center')
        
    sns.regplot(x = "lnGDP_pc", y = "religion_C2", data = df_fig3, ax = ax[2][1])
    ax[2][1].set_ylabel("Religion $\hat{S}$", size = 14)
    ax[2][1].set_xlabel("Log per capita GDP", size = 14)
    for i in range(df_fig3.shape[0]):
        ax[2][1].annotate(df_fig3.country[i],
                         (df_fig3.lnGDP_pc[i],df_fig3.religion_C2[i]),
                         ha = 'left',
                         va = 'center')
    
    return plt.show()



def ext1(df) :
    var_color_dict = {'ethnicity_C2': 'blue',
                      'language_C2': 'red', 
                      'religion_C2': 'yellow'}
    for var in var_color_dict:
        fig = sns.histplot(df[var],                  
                           color = var_color_dict[var], 
                           label = None,
                           kde = True)
    fig.legend(['Ethnicity','Language','Religion'],title = 'Segregation')
    fig.set_xlabel('')
    
    return plt.show()

    
def ext3(df):
    
    df['democ1'] = ["More" if i >= 1 else "Less" for i in df['democ']]

    list1 = {'ethnicity_C2':'Ethnicity',
             'language_C2':'Language',
             'religion_C2':'Religion'}

    for name in list1:
        fig = sns.lmplot(x='lnGDP_pc', y = f'{name}',
                         hue = 'democ1', data = df, ci = None,
                         hue_order = ['More','Less'],
                         markers = ['o','x'])
        fig.set_axis_labels('Log per capita GDP', list1[name])
    
    return plt.show()

def ext4(df):
    reg = sm.OLS(df['RulLaw'],sm.add_constant(df[['ethnicity_C2','ethnicity_I','lnpopulation','lnGDP_pc','protestants',
                                 'muslims','catholics','latitude','LOEnglish','LOGerman','LOSocialist',
                                 'LOScandin','democ','mtnall']])).fit()
    fig = plt.figure(figsize=(12,8))
    fig = sm.graphics.plot_regress_exog(reg, 'ethnicity_C2', fig=fig)
    
    return plt.show()



def ext5(df, world) :
    
    pd.options.mode.chained_assignment = None
    
    df['country'] = df['country'].str.capitalize()
    df['country'] = df['country'].str.replace(pat = '_', repl = ' ', regex=False)
    df['country'] = df['country'].replace(['Usa','United kingdom','Korea','Saudi arabia','Dominican republic','Newzealand'
                                             ,'Central african republic','burkina_faso','Czech republic'],
                                            ['United States of America','United Kingdom','South Korea','Saudi Arabia',
                                             'Dominican Rep.','New Zealand','Central African Rep.',
                                             'Burkina Faso','Czechia'])
    df.rename(columns = {'country' : 'name'}, inplace = True)
    
    world1 = pd.merge(df[['name','democ']],world, how = 'outer', on = 'name')
    world1 = gpd.GeoDataFrame(world1)
    
    world1.plot(column = 'democ',
                legend = True, figsize=(15, 6), cmap ='plasma',
                missing_kwds={'color': 'lightgrey',
                              'edgecolor': 'black',
                              'hatch': '///',
                              'label': 'Missing values'})
                
    return


def ext6(df):
    fig = plt.figure(figsize=(13,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    fig1 = probplot(df['lnGDP_pc'].dropna(), plot = ax1)
    ax1.title.set_text('lnGDP per capita')
    fig2 = probplot(df['democ'].dropna(), plot = ax2)
    ax2.title.set_text('Democ')
    return
