import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from stargazer.stargazer import Stargazer
from IPython.core.display import HTML
from IPython.core.interactiveshell import InteractiveShell
from statsmodels.sandbox.regression.gmm import IV2SLS



def create_table1(df):
    #Ethnicity table
    df_1 = df.sort_values(by="ethnicity_C2", ascending=False)
    df_1 = df_1[['country', 'ethnicity_C2', 'ethnicity_I']].head()
    df_1 = df_1.reset_index(drop=True)

    df_2 = df.sort_values(by="ethnicity_C2", ascending=True)
    df_2 = df_2[['country', 'ethnicity_C2', 'ethnicity_I']].head()
    df_2 = df_2.reset_index(drop=True)

    df_eth = pd.concat([df_1,df_2], axis = 1)

    df_eth = df_eth.round(decimals=3)
    #Language table
    df_3 = df.sort_values(by="language_C2", ascending=False)
    df_3 = df_3[['country', 'language_C2', 'language_I']].head()
    df_3 = df_3.reset_index(drop=True)

    df_4 = df.sort_values(by="language_C2", ascending=True)
    df_4 = df_4[['country', 'language_C2', 'language_I']].head()
    df_4 = df_4.reset_index(drop=True)

    df_lang = pd.concat([df_3,df_4], axis = 1)

    df_lang = df_lang.round(decimals=3)
    #Religion table
    df_5 = df.sort_values(by="religion_C2", ascending=False)
    df_5 = df_5[['country', 'religion_C2', 'religion_I']].head()
    df_5 = df_5.reset_index(drop=True)

    df_6 = df.sort_values(by="religion_C2", ascending=True)
    df_6 = df_6[['country', 'religion_C2', 'religion_I']].head()
    df_6 = df_6.reset_index(drop=True)

    df_rel = pd.concat([df_5,df_6], axis = 1)

    df_rel = df_rel.round(decimals=3)

    #Rename rows and columns of complete table
    table1 = pd.concat([df_eth,df_lang,df_rel],axis = 1)
    table1.columns = pd.MultiIndex.from_product([['Ethinicity','Language','Religion'],
                                                 ['Most segregated','Least segregated',],
                                                 ['Country','$\hat{S}$', '$F$']])
    return table1



def create_table2(df):
    
    variables = df[['ethnicity_C2','language_C2','religion_C2','ethnicity_C','language_C','religion_C',
                   'voice','PolStab','GovEffec','RegQual','RulLaw','ConCorr']]
    table2 = pd.DataFrame()
    
    table2 = variables.corr()
    
    table2.drop(['voice','PolStab','GovEffec','RegQual','RulLaw','ConCorr'], axis = 1, inplace = True)
    
    table2.drop(['ethnicity_C2','ethnicity_C','language_C2','language_C','religion_C2','religion_C'], 
                axis = 0, inplace = True)
    
    table2 = table2.round(decimals=2)
    
    table2.columns = pd.MultiIndex.from_product([['Segregation indicies'],
                                                 ['Ethnicity $\hat{S}$','Language $\hat{S}$', 'Religion $\hat{S}$',
                                                  'Ethnicity $\tilde{S}$','Language $\tilde{S}$', 'Religion $\tilde{S}$']]
                                               )
    table2 = table2.rename(index = {
                                    "voice" : 'Voice',
                                    "Polstab" : 'Political stability',
                                    "GovEffec" : 'Government effectiveness',
                                    "RegQual" : 'Regulatory quality',
                                    "RulLaw" : 'Rule fo law',
                                    "ConCorr" : 'Control of corruption'}
                          )
    
    return table2



def table3_7(df, regression_type) :

    
    df_3_7E = df[['ethnicity_C2','ethnicity_instrument_C2_thresh','ethnicity_I','lnpopulation','lnGDP_pc',
                     'protestants','muslims','catholics','latitude','LOEnglish','LOGerman','LOSocialist','lnArea',
                     'LOScandin','democ','mtnall','RulLaw']].dropna(axis=0)
    df_3_7L = df[['language_C2','language_instrument_C2_thresh','language_I','lnpopulation','lnGDP_pc',
                     'protestants','muslims','catholics','latitude','LOEnglish','LOGerman','LOSocialist','lnArea',
                     'LOScandin','democ','mtnall','RulLaw']].dropna(axis=0)
    df_3_7R = df[['religion_C2','religion_instrument_C2_thresh','religion_I','lnpopulation','lnGDP_pc',
                     'protestants','muslims','catholics','latitude','LOEnglish','LOGerman','LOSocialist','lnArea',
                     'LOScandin','democ','mtnall','RulLaw']].dropna(axis=0)
    
    exo = sm.add_constant(df_3_7E[['ethnicity_C2','ethnicity_I','lnpopulation','lnGDP_pc','protestants','muslims',
                                      'catholics','latitude','LOEnglish','LOGerman','LOSocialist', 'LOScandin','lnArea',
                                      'democ','mtnall']])
    exo2 = sm.add_constant(df_3_7E[['ethnicity_C2','ethnicity_I']])
    exo3 = sm.add_constant(df_3_7L[['language_C2','language_I','lnpopulation','lnGDP_pc','protestants','lnArea',
                                       'muslims','catholics','latitude','LOEnglish','LOGerman','LOSocialist', 'LOScandin',
                                       'democ','mtnall']])
    exo4 = sm.add_constant(df_3_7L[['language_C2','language_I']])
    exo5 = sm.add_constant(df_3_7R[['religion_C2','religion_I','lnpopulation','lnGDP_pc','protestants',
                                       'muslims','catholics','latitude','LOEnglish','LOGerman','LOSocialist','lnArea',
                                       'democ','mtnall']])
    exo6 = sm.add_constant(df_3_7R[['religion_C2','religion_I']])
    
    if regression_type == 'IV2SLS' :

        reg = IV2SLS(df_3_7E['RulLaw'],
                     exo,
                     sm.add_constant(df_3_7E[['ethnicity_instrument_C2_thresh','ethnicity_I','lnpopulation','lnGDP_pc',
                                                 'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                                 'LOSocialist', 'LOScandin','democ','mtnall','lnArea']])).fit()
        reg2 = IV2SLS(df_3_7E['RulLaw'],
                      exo2,
                      sm.add_constant(df_3_7E[['ethnicity_instrument_C2_thresh',
                                                  'ethnicity_I']])).fit()
        reg3 = IV2SLS(df_3_7L['RulLaw'],
                      exo3,
                      sm.add_constant(df_3_7L[['language_instrument_C2_thresh','language_I','lnpopulation','lnGDP_pc',
                                                  'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                                  'LOSocialist', 'LOScandin','democ','mtnall','lnArea']])).fit()
        reg4 = IV2SLS(df_3_7L['RulLaw'],
                      exo4,
                      sm.add_constant(df_3_7L[['language_instrument_C2_thresh',
                                                  'language_I']])).fit()
        reg5 = IV2SLS(df_3_7R['RulLaw'],
                      exo5,
                      sm.add_constant(df_3_7R[['religion_instrument_C2_thresh','religion_I','lnpopulation','lnGDP_pc',
                                                  'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                                  'LOSocialist','democ','mtnall','lnArea']])).fit()
        reg6 = IV2SLS(df_3_7R['RulLaw'],
                      exo6,
                      sm.add_constant(df_3_7R[['religion_instrument_C2_thresh',
                                                  'religion_I']])).fit()
    elif regression_type == 'OLS' : 
        reg2 = sm.OLS(df_3_7E['RulLaw'], exo2).fit(cov_type = 'HC1')
        reg = sm.OLS(df_3_7E['RulLaw'], exo).fit(cov_type = 'HC1')
        reg4 = sm.OLS(df_3_7L['RulLaw'], exo4).fit(cov_type = 'HC1')
        reg3 = sm.OLS(df_3_7L['RulLaw'], exo3).fit(cov_type = 'HC1')
        reg6 = sm.OLS(df_3_7R['RulLaw'], exo6).fit(cov_type = 'HC1')
        reg5 = sm.OLS(df_3_7R['RulLaw'], exo5).fit(cov_type = 'HC1')

    stargazer = Stargazer([reg2, reg, reg4, reg3, reg6, reg5])
    stargazer.covariate_order(['ethnicity_C2', 'ethnicity_I','language_C2','language_I','religion_C2','religion_I',
                               'lnpopulation','lnGDP_pc','lnArea','protestants','muslims','catholics',
                               'latitude','LOEnglish','LOGerman','LOSocialist','LOScandin','democ','mtnall','const'])
    stargazer.rename_covariates({'ethnicity_C2' : 'Segregation $\hat{S}$ (ethnicity)',
                               'ethnicity_I'  : 'Fractionalization $F$ (ethnicity)',
                               'language_C2' : 'Segregation $\hat{S}$ (language)',
                               'language_I'  : 'Fractionalization $F$ (language)',
                               'religion_C2' : 'Segregation $\hat{S}$ (religion)',
                               'religion_I'  : 'Fractionalization $F$ (religion)',
                               'lnpopulation' : 'ln (population)',
                               'lnGDP_pc' : 'ln (GDP per capita)',
                               'lnArea' : 'ln (average size of region)',
                               'protestants' : 'Pretestants share',
                               'muslims' : 'Muslmis Share',
                               'catholics' : 'Catholics share',
                               'latitude' : 'Latitude',
                               'LOEnglish' : 'English legal origin',
                               'LOGerman' : 'German legal origin',
                               'LOSocialist' : 'Socialist legal origin',
                               'LOScandin' : 'Scandinavian legal origin',
                               'democ' : 'Democratic tradition',
                               'mtnall' : 'Mountains',
                               'const' : 'Constant'})
    return HTML(stargazer.render_html())



def table4_5(df,name) :    

    df_table4A = df[[f'{name}_C2',f'{name}_I','lnpopulation','lnGDP_pc','protestants','muslims',
                    'catholics','latitude','LOEnglish','LOGerman','LOSocialist','LOScandin','democ',
                    'mtnall','voice','PolStab','GovEffec','RegQual','ConCorr','RulLaw']].dropna(axis = 0)
    
    df_table4B = df_table4A[[f'{name}_C2',f'{name}_I','voice','PolStab','GovEffec','RegQual','ConCorr','RulLaw']]
    
    df_table4C = df_table4A[df_table4A.democ > 1]
    
    xA = sm.add_constant(df_table4A[[f'{name}_C2',f'{name}_I','lnpopulation','lnGDP_pc','protestants',
                                       'muslims','catholics','latitude','LOEnglish','LOGerman','LOSocialist',
                                       'LOScandin','democ','mtnall']])

    xB = sm.add_constant(df_table4B[[f'{name}_C2', f'{name}_I']])

    xC = sm.add_constant(df_table4C[[f'{name}_C2',f'{name}_I','lnpopulation','lnGDP_pc','protestants',
                                       'muslims','catholics','latitude','LOEnglish','LOGerman','LOSocialist',
                                       'LOScandin','democ','mtnall']])
    df_table4s = [df_table4A, df_table4B, df_table4C]

    xs = [xA, xB, xC]

    
    y = [
        [f'y{idx}A', f'y{idx}B', f'y{idx}C']
        for idx in range(1, 7)
        ]
    est = [
        [f'est{idx}A', f'est{idx}B', f'est{idx}C']
        for idx in range(1, 7)
        ]
    

    star = ['starA','starB','starC']

    for idx, i in enumerate(['A','B','C']) :

        y[0][idx] = df_table4s[idx]['voice']
        y[1][idx] = df_table4s[idx]['PolStab']
        y[2][idx] = df_table4s[idx]['GovEffec']
        y[3][idx] = df_table4s[idx]['RegQual']
        y[4][idx] = df_table4s[idx]['RulLaw']
        y[5][idx] = df_table4s[idx]['ConCorr']

        est[0][idx] = sm.OLS(y[0][idx], xs[idx]).fit(cov_type = 'HC1')
        est[1][idx] = sm.OLS(y[1][idx], xs[idx]).fit(cov_type = 'HC1')
        est[2][idx] = sm.OLS(y[2][idx], xs[idx]).fit(cov_type = 'HC1')
        est[3][idx] = sm.OLS(y[3][idx], xs[idx]).fit(cov_type = 'HC1')
        est[4][idx] = sm.OLS(y[4][idx], xs[idx]).fit(cov_type = 'HC1')
        est[5][idx] = sm.OLS(y[5][idx], xs[idx]).fit(cov_type = 'HC1')

        star[idx] = Stargazer([est[0][idx],est[1][idx],est[2][idx],est[3][idx],est[4][idx],est[5][idx]])
    for i in range(3) :
        star[i].covariate_order([f'{name}_C2',f'{name}_I'])
        star[i].rename_covariates({f'{name}_C2' : 'Segregation $\hat{S}$ ('f'{name}'')',
                                   f'{name}_I'  : 'Fractionalization $F$ ('f'{name}'')'})
        star[i].show_model_numbers(False)
        star[i].custom_columns(['Voice', 
                                'Political stability', 
                                'Govern-t effectiv.', 
                                'Regul. quality',
                                'Rule of law',
                                'Control of corr'],
                                [1,1,1,1,1,1])
    star[0].add_line('Controls', ['Yes','Yes','Yes','Yes','Yes','Yes'])
    star[0].add_line('Sample', ['Full','Full','Full','Full','Full','Full'])
    star[1].add_line('Controls', ['No','No','No','No','No','No'])
    star[1].add_line('Sample', ['Full','Full','Full','Full','Full','Full'])
    star[2].add_line('Controls', ['Yes','Yes','Yes','Yes','Yes','Yes'])
    star[2].add_line('Sample', ['Democ','Democ','Democ','Democ','Democ','Democ'])
            
            
    star[0].title('Panel A. Baseline : All controls and full sample')
    star[1].title('Panel B. No controls and full sample')
    star[2].title('Panel C. All controls; sample excludes dictatorship')

    return [star[0],star[1],star[2]]



def table6(df, alternative = True) :    

    df_6E = df[['ethnicity_C2', 'ethnicity_I','ethnicity_C','ethnicity_instrument_C_thresh',
                     'ethnicity_instrument_C2_thresh','lnpopulation','lnGDP_pc','protestants','muslims',
                     'catholics','latitude','LOEnglish','LOGerman','LOSocialist','LOScandin','democ',
                     'mtnall','RulLaw','country']].dropna(axis=0)
    df_6L = df[['language_C2', 'language_I','language_C','language_instrument_C_thresh',
                     'language_instrument_C2_thresh','lnpopulation','lnGDP_pc','protestants','muslims',
                     'catholics','latitude','LOEnglish','LOGerman','LOSocialist','LOScandin','democ',
                     'mtnall','RulLaw','country']].dropna(axis=0)
    df_6R = df[['religion_C2', 'religion_I','religion_C','religion_instrument_C_thresh',
                     'religion_instrument_C2_thresh','lnpopulation','lnGDP_pc','protestants','muslims',
                     'catholics','latitude','LOEnglish','LOGerman','LOSocialist','LOScandin','democ',
                     'mtnall','RulLaw','country']].dropna(axis=0)
    
    df_6E_demo = df_6E[df_6E.democ >= 1]
    df_6L_demo = df_6L[df_6L.democ >= 1]
    df_6R_demo = df_6R[df_6R.democ >= 1]
    
    x1 = sm.add_constant(df_6E[['ethnicity_instrument_C2_thresh', 'ethnicity_I','lnpopulation','lnGDP_pc',
                                     'protestants','muslims','catholics','latitude','LOEnglish',
                                     'LOGerman','LOSocialist','LOScandin','democ','mtnall']])
    x2 = sm.add_constant(df_6L[['language_instrument_C2_thresh', 'language_I','lnpopulation','lnGDP_pc',
                                     'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                     'LOSocialist','LOScandin','democ','mtnall']])
    x3 = sm.add_constant(df_6R[['religion_instrument_C2_thresh', 'religion_I','lnpopulation','lnGDP_pc',
                                     'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                     'LOSocialist','democ','mtnall']])
    x4 = sm.add_constant(df_6E_demo[['ethnicity_instrument_C2_thresh', 'ethnicity_I','lnpopulation','lnGDP_pc',
                                          'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                          'LOSocialist','LOScandin','democ','mtnall']])
    x5 = sm.add_constant(df_6L_demo[['language_instrument_C2_thresh', 'language_I','lnpopulation','lnGDP_pc',
                                          'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                          'LOSocialist','LOScandin','democ','mtnall']])
    x6 = sm.add_constant(df_6R_demo[['religion_instrument_C2_thresh', 'religion_I','lnpopulation','lnGDP_pc',
                                          'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                          'LOSocialist','democ','mtnall']])
    
    y1 = df_6E['ethnicity_C2']
    y2 = df_6L['language_C2']
    y3 = df_6R['religion_C2']
    y4 = df_6E_demo['ethnicity_C2']
    y5 = df_6L_demo['language_C2']
    y6 = df_6R_demo['religion_C2']
    
    est1 = sm.OLS(y1, x1).fit(cov_type = 'HC1')
    est2 = sm.OLS(y2, x2).fit(cov_type = 'HC1')
    est3 = sm.OLS(y3, x3).fit(cov_type = 'HC1')
    est4 = sm.OLS(y4, x4).fit(cov_type = 'HC1')
    est5 = sm.OLS(y5, x5).fit(cov_type = 'HC1')
    est6 = sm.OLS(y6, x6).fit(cov_type = 'HC1')
    
    x1a = sm.add_constant(df_6E[['ethnicity_instrument_C_thresh', 'ethnicity_I','lnpopulation','lnGDP_pc',
                                      'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                      'LOSocialist','LOScandin','democ','mtnall']])
    x2a = sm.add_constant(df_6L[['language_instrument_C_thresh', 'language_I','lnpopulation','lnGDP_pc',
                                      'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                      'LOSocialist','LOScandin','democ','mtnall']])
    x3a = sm.add_constant(df_6R[['religion_instrument_C_thresh', 'religion_I','lnpopulation','lnGDP_pc',
                                      'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                      'LOSocialist','democ','mtnall']])
    x4a = sm.add_constant(df_6E_demo[['ethnicity_instrument_C_thresh', 'ethnicity_I','lnpopulation','lnGDP_pc',
                                           'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                           'LOSocialist','LOScandin','democ','mtnall']])
    x5a = sm.add_constant(df_6L_demo[['language_instrument_C_thresh', 'language_I','lnpopulation','lnGDP_pc',
                                           'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                           'LOSocialist','LOScandin','democ','mtnall']])
    x6a = sm.add_constant(df_6R_demo[['religion_instrument_C_thresh', 'religion_I','lnpopulation','lnGDP_pc',
                                           'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                           'LOSocialist','democ','mtnall']])
    
    y1a = df_6E['ethnicity_C']
    y2a = df_6L['language_C']
    y3a = df_6R['religion_C']
    y4a = df_6E_demo['ethnicity_C']
    y5a = df_6L_demo['language_C']
    y6a = df_6R_demo['religion_C']

    est1a = sm.OLS(y1a, x1a).fit(cov_type = 'HC1')
    est2a = sm.OLS(y2a, x2a).fit(cov_type = 'HC1')
    est3a = sm.OLS(y3a, x3a).fit(cov_type = 'HC1')
    est4a = sm.OLS(y4a, x4a).fit(cov_type = 'HC1')
    est5a = sm.OLS(y5a, x5a).fit(cov_type = 'HC1')
    est6a = sm.OLS(y6a, x6a).fit(cov_type = 'HC1')
    
    
    df_6Lb = df_6L.set_index('country')
    df_6Lb_demo = df_6L_demo.set_index('country')
    
    x2b = sm.add_constant(df_6Lb[['language_instrument_C_thresh', 'language_I','lnpopulation','lnGDP_pc',
                                      'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                      'LOSocialist','LOScandin','democ','mtnall']].drop(index = 'usa'))

    x5b = sm.add_constant(df_6Lb_demo[['language_instrument_C_thresh', 'language_I','lnpopulation','lnGDP_pc',
                                           'protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                                           'LOSocialist','LOScandin','democ','mtnall']].drop(index = 'usa'))
    y2b = df_6Lb['language_C'].drop(index = 'usa')
    y5b = df_6Lb_demo['language_C'].drop(index = 'usa')

    est2b = sm.OLS(y2b, x2b).fit(cov_type = 'HC1')
    est5b = sm.OLS(y5b, x5b).fit(cov_type = 'HC1')
    
    
    stargazer = Stargazer([est1, est2, est3, est4, est5, est6])
    stargazer_a = Stargazer([est1a, est2a, est3a, est4a, est5a, est6a])
    stargazer_b = Stargazer([est2b, est5b])
    
    stargazer.covariate_order(['ethnicity_instrument_C2_thresh', 'ethnicity_I',
                               'language_instrument_C2_thresh', 'language_I',
                               'religion_instrument_C2_thresh', 'religion_I'])
    stargazer.rename_covariates({'ethnicity_instrument_C2_thresh':'Instrument E',
                                 'ethnicity_I':'$F$ (ethnicity)',
                                 'language_instrument_C2_thresh':'Instrument L',
                                 'language_I':'$F$ (language)',
                                 'religion_instrument_C2_thresh':'Instrument R',
                                 'religion_I':'$F$ (religion)'
                                  })
    stargazer.custom_columns(['E$\hat{S}$', 
                              'L$\hat{S}$',
                              'R$\hat{S}$', 
                              'E$\hat{S}$',
                              'L$\hat{S}$', 
                              'R$\hat{S}$'],
                              [1,1,1,1,1,1])
    stargazer.show_model_numbers(False)
    stargazer.add_line('Sample', ['Full','Full','Full','Democracy','Democracy','Democracy'])
    stargazer.title('Panel A. Segregation index $\hat{S}$')
    
    stargazer_a.covariate_order(['ethnicity_instrument_C_thresh', 'ethnicity_I',
                               'language_instrument_C_thresh', 'language_I',
                               'religion_instrument_C_thresh', 'religion_I'])
    stargazer_a.rename_covariates({'ethnicity_instrument_C_thresh':'Instrument E',
                                 'ethnicity_I':'$F$ (ethnicity)',
                                 'language_instrument_C_thresh':'Instrument L',
                                 'language_I':'$F$ (language)',
                                 'religion_instrument_C_thresh':'Instrument R',
                                 'religion_I':'$F$ (religion)'
                                  })
    stargazer_a.custom_columns(['E$\\tilde{S}$', 
                                'L$\\tilde{S}$',
                                'R$\\tilde{S}$', 
                                'E$\\tilde{S}$',
                                'L$\\tilde{S}$', 
                                'R$\\tilde{S}$'],
                                [1,1,1,1,1,1])
    stargazer_a.show_model_numbers(False)
    stargazer_a.add_line('Sample', ['Full','Full','Full','Democracy','Democracy','Democracy'])
    stargazer_a.title('Panel B. Segregation index $\\tilde{S}$')
    
    stargazer_b.covariate_order(['language_instrument_C_thresh', 'language_I'])
    stargazer_b.rename_covariates({'language_instrument_C_thresh':'Instrument L',
                                   'language_I':'$F$ (language)'
                                    })
    stargazer_b.custom_columns(['L$\\tilde{S}$',
                                'L$\\tilde{S}$'],
                                [1,1])
    stargazer_b.show_model_numbers(False)
    stargazer_b.add_line('Sample', ['Full','Democracy'])
    stargazer_b.title('Panel C. Segregation index $\\tilde{S}$ for language with sample excluding the US')
    
    return [stargazer,stargazer_a,stargazer_b]



def table8_9_ext7(df,name,GDP) :    

    df_8_9A = df[[f'{name}_C2',f'{name}_I',f'{name}_instrument_C2_thresh','lnpopulation','lnGDP_pc','protestants','muslims',
                     'catholics','latitude','LOEnglish','LOGerman','LOSocialist','LOScandin','democ',
                     'mtnall','voice','PolStab','GovEffec','RegQual','ConCorr','RulLaw'
                   ]].dropna(axis = 0)
    df_8_9B = df_8_9A[[f'{name}_C2',f'{name}_instrument_C2_thresh',f'{name}_I','voice','PolStab','GovEffec','RegQual',
                             'ConCorr','RulLaw']]
    if GDP == 'democ':
        df_8_9C = df_8_9A[df_8_9A.democ >= 1]
    elif GDP == 'GDP':
        df_8_9C = df_8_9A[df_8_9A.lnGDP_pc >= 7]
    
    exoA = sm.add_constant(df_8_9A[[f'{name}_C2',f'{name}_I','lnpopulation','lnGDP_pc','protestants',
                                       'muslims','catholics','latitude','LOEnglish','LOGerman','LOSocialist',
                                       'LOScandin','democ','mtnall']])

    exoB = sm.add_constant(df_8_9B[[f'{name}_C2', f'{name}_I']])

    exoC = sm.add_constant(df_8_9C[[f'{name}_C2',f'{name}_I','lnpopulation','lnGDP_pc','protestants',
                                       'muslims','catholics','latitude','LOEnglish','LOGerman','LOSocialist',
                                       'LOScandin','democ','mtnall']])
        
    insA = sm.add_constant(df_8_9A[[f'{name}_instrument_C2_thresh',f'{name}_I','lnpopulation','lnGDP_pc','protestants',
                                       'muslims','catholics','latitude','LOEnglish','LOGerman','LOSocialist',
                                       'LOScandin','democ','mtnall']])

    insB = sm.add_constant(df_8_9B[[f'{name}_instrument_C2_thresh', f'{name}_I']])

    insC = sm.add_constant(df_8_9C[[f'{name}_instrument_C2_thresh',f'{name}_I','lnpopulation','lnGDP_pc','protestants',
                                       'muslims','catholics','latitude','LOEnglish','LOGerman','LOSocialist',
                                       'LOScandin','democ','mtnall']])
    
    df_8_9s = [df_8_9A, df_8_9B, df_8_9C]

    exos = [exoA, exoB, exoC]
    
    inss = [insA, insB, insC]

    
    y = [
        [f'y{idx}A', f'y{idx}B', f'y{idx}C']
        for idx in range(1, 7)
        ]
    est = [
        [f'est{idx}A', f'est{idx}B', f'est{idx}C']
        for idx in range(1, 7)
        ]
    

    star = ['starA','starB','starC']

    for idx, i in enumerate(['A','B','C']) :

        y[0][idx] = df_8_9s[idx]['voice']
        y[1][idx] = df_8_9s[idx]['PolStab']
        y[2][idx] = df_8_9s[idx]['GovEffec']
        y[3][idx] = df_8_9s[idx]['RegQual']
        y[4][idx] = df_8_9s[idx]['RulLaw']
        y[5][idx] = df_8_9s[idx]['ConCorr']

        est[0][idx] = IV2SLS(y[0][idx], exos[idx], inss[idx]).fit()
        est[1][idx] = IV2SLS(y[1][idx], exos[idx], inss[idx]).fit()
        est[2][idx] = IV2SLS(y[2][idx], exos[idx], inss[idx]).fit()
        est[3][idx] = IV2SLS(y[3][idx], exos[idx], inss[idx]).fit()
        est[4][idx] = IV2SLS(y[4][idx], exos[idx], inss[idx]).fit()
        est[5][idx] = IV2SLS(y[5][idx], exos[idx], inss[idx]).fit()
        
        star[idx] = Stargazer([est[0][idx],est[1][idx],est[2][idx],est[3][idx],est[4][idx],est[5][idx]])
    for i in range(3) :
        star[i].covariate_order([f'{name}_C2',f'{name}_I'])
        star[i].rename_covariates({f'{name}_C2' : 'Segregation $\hat{S}$ ('f'{name}'')',
                                   f'{name}_I'  : 'Fractionalization $F$ ('f'{name}'')'})
        star[i].show_model_numbers(False)
        star[i].custom_columns(['Voice', 
                                'Political stability', 
                                'Govern-t effectiv.', 
                                'Regul. quality',
                                'Rule of law',
                                'Control of corr'],
                                [1,1,1,1,1,1])
    if GDP == 'democ' :
        star[0].add_line('Controls', ['Yes','Yes','Yes','Yes','Yes','Yes'])
        star[0].add_line('Sample', ['Full','Full','Full','Full','Full','Full'])
        star[1].add_line('Controls', ['No','No','No','No','No','No'])
        star[1].add_line('Sample', ['Full','Full','Full','Full','Full','Full'])
        star[2].add_line('Controls', ['Yes','Yes','Yes','Yes','Yes','Yes'])
        star[2].add_line('Sample', ['Democ','Democ','Democ','Democ','Democ','Democ'])
        
        star[0].title('Panel A. Baseline : All controls and full sample')
        star[1].title('Panel B. No controls and full sample')
        star[2].title('Panel C. All controls; sample excludes dictatorship')

        return [star[0],star[1],star[2]]
    
    if GDP == 'GDP' :
        if name == 'ethnicity' :
            star[2].title('Panal A. Ethnicity: All controls; sample excludes poorest countries')
        elif name == 'language' :
            star[2].title('Panel B. Language: All controls; sample excludes poorest countries')
        return star[2]



def table10_11(df,name,democ) :
    
    full_x = [f'{name}_I',f'{name}_C2','lnpopulation','lnGDP_pc','protestants','muslims',
             'catholics','latitude','LOEnglish','LOGerman','LOSocialist','LOScandin','democ',
             'mtnall']
    ins = [f'{name}_I',f'{name}_instrument_C2_thresh','lnpopulation','lnGDP_pc','protestants','muslims',
              'catholics','latitude','LOEnglish','LOGerman','LOSocialist','LOScandin','democ',
              'mtnall']
    
    df_10_11_1 = df[[f'{name}_C2', f'{name}_I',
                     f'{name}_instrument_C2_thresh','lnpopulation','lnGDP_pc','protestants','muslims',
                     'catholics','latitude','LOEnglish','LOGerman','LOSocialist','LOScandin','democ',
                     'mtnall','icrg_qog']].dropna(axis=0)
    df_10_11_2 = df[[f'{name}_C2', f'{name}_I',
                     f'{name}_instrument_C2_thresh','lnpopulation','lnGDP_pc','protestants','muslims',
                     'catholics','latitude','LOEnglish','LOGerman','LOSocialist','LOScandin','democ',
                     'mtnall','ef_regul','ef_corruption','ef_property_rights']].dropna(axis=0)
    df_10_11_3 = df[[f'{name}_C2', f'{name}_I',
                     f'{name}_instrument_C2_thresh','lnpopulation','lnGDP_pc','protestants','muslims',
                     'catholics','latitude','LOEnglish','LOGerman','LOSocialist','LOScandin','democ',
                     'mtnall','taxevas']].dropna(axis=0)

    if democ == 'democracy' :
        df_10_11_1 = df_10_11_1[df_10_11_1.democ >= 1]
        df_10_11_2 = df_10_11_2[df_10_11_2.democ >= 1]
        df_10_11_3 = df_10_11_3[df_10_11_3.democ >= 1]
        
        x1 = sm.add_constant(df_10_11_1[full_x])
        x2 = sm.add_constant(df_10_11_2[full_x])
        x3 = sm.add_constant(df_10_11_3[full_x])

        ins1 = sm.add_constant(df_10_11_1[ins])
        ins2 = sm.add_constant(df_10_11_2[ins])
        ins3 = sm.add_constant(df_10_11_3[ins])
        
    else :
        x1 = sm.add_constant(df_10_11_1[[f'{name}_I',f'{name}_C2']])
        x2 = sm.add_constant(df_10_11_2[[f'{name}_I',f'{name}_C2']])
        x3 = sm.add_constant(df_10_11_3[[f'{name}_I',f'{name}_C2']])

        ins1 = sm.add_constant(df_10_11_1[[f'{name}_I',f'{name}_instrument_C2_thresh']])
        ins2 = sm.add_constant(df_10_11_2[[f'{name}_I',f'{name}_instrument_C2_thresh']])
        ins3 = sm.add_constant(df_10_11_3[[f'{name}_I',f'{name}_instrument_C2_thresh']])
            
    y1 = df_10_11_1['icrg_qog']
    y2 = df_10_11_2['ef_corruption']
    y3 = df_10_11_2['ef_property_rights']
    y4 = df_10_11_2['ef_regul']
    y5 = df_10_11_3['taxevas']

    est1 = sm.OLS(y1,x1).fit(cov_type = 'HC1')
    est2 = IV2SLS(y1, x1,ins1).fit()
    est3 = sm.OLS(y2, x2).fit(cov_type = 'HC1')
    est4 = IV2SLS(y2, x2 ,ins2).fit()
    est5 = sm.OLS(y3, x2).fit(cov_type = 'HC1')
    est6 = IV2SLS(y3, x2 ,ins2).fit()
    est7 = sm.OLS(y4, x2).fit(cov_type = 'HC1')
    est8 = IV2SLS(y4, x2, ins2).fit()
    est9 = sm.OLS(y5, x3).fit(cov_type = 'HC1')
    est10 = IV2SLS(y5, x3, ins3).fit()

    stargazer = Stargazer([est1,est2,est3,est4,est5,est6,est7,est8,est9,est10])
    stargazer.custom_columns(['ICRG quality of gov','EF Corruption','EF Property rights',
                              'EF Regulation','Tax eva'],[2,2,2,2,2])
    stargazer.show_model_numbers(False)
    stargazer.covariate_order([f'{name}_C2',f'{name}_I'])
    stargazer.rename_covariates({f'{name}_C2' : 'Segregation $\hat{S}$ ('f'{name}'')',
                                 f'{name}_I'  : 'Fractionalization $F$ ('f'{name}'')'})
    stargazer.add_line('Method', ['OLS','2SLS','OLS','2SLS','OLS','2SLS','OLS','2SLS','OLS','2SLS'])
    
    if democ == 'democracy':
        stargazer.title('Panel B. Democracies sample, all controls')
        return stargazer

    else:
        stargazer.title('Panel A. Full sample, no additional controls')
        return stargazer

    
    
def df_table12(df,name) :    
    df_table12 = df[[f'{name}_C2',f'{name}_instrument_C2_thresh',f'{name}_I','trust','democ','lnpopulation','lnArea',
                     'lnGDP_pc','protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                     'LOSocialist','LOScandin','mtnall']].dropna(axis=0)

    df_demo = df_table12[df_table12.democ > 1]

    dep1 = df_table12['trust']
    dep2 = df_demo['trust']

    exo1 = sm.add_constant(df_table12[f'{name}_C2'])
    exo2 =  sm.add_constant(df_table12[[f'{name}_C2', f'{name}_I','lnpopulation','lnArea',
                     'lnGDP_pc','protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                     'LOSocialist','LOScandin','democ','mtnall']])
    exo3 =  sm.add_constant(df_demo[[f'{name}_C2',f'{name}_I','lnpopulation','lnArea',
                     'lnGDP_pc','protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                     'LOSocialist','LOScandin','democ','mtnall']])

    ins1 = sm.add_constant(df_table12[f'{name}_instrument_C2_thresh'])
    ins2 = sm.add_constant(df_table12[[f'{name}_instrument_C2_thresh',f'{name}_I','lnpopulation','lnArea',
                     'lnGDP_pc','protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                     'LOSocialist','LOScandin','democ','mtnall']])
    ins3 = sm.add_constant(df_demo[[f'{name}_instrument_C2_thresh',f'{name}_I','lnpopulation','lnArea',
                     'lnGDP_pc','protestants','muslims','catholics','latitude','LOEnglish','LOGerman',
                     'LOSocialist','LOScandin','democ','mtnall']])
    
    reg1 = sm.OLS(dep1,exo1).fit(cov_type = 'HC1')
    reg2 = sm.OLS(dep1,exo2).fit(cov_type = 'HC1')
    reg3 = sm.OLS(dep2,exo3).fit(cov_type = 'HC1')
    reg4 = IV2SLS(dep1,exo1,ins1).fit()
    reg5 = IV2SLS(dep1,exo2,ins2).fit()
    reg6 = IV2SLS(dep2,exo3,ins3).fit()

    stargazer = Stargazer([reg1,reg2,reg3,reg4,reg5,reg6])
    stargazer.covariate_order([f'{name}_C2',f'{name}_I'])
    stargazer.rename_covariates({f'{name}_C2' : 'Segregation $\hat{S}$ ('f'{name}'')',
                                 f'{name}_I'  : 'Fractionalization $F$ ('f'{name}'')'})
    
    stargazer.custom_columns(['OLS','OLS','OLS','2SLS','2SLS','2SLS'],[1,1,1,1,1,1])
    stargazer.add_line('Controls', ['No','Yes','Yes','No','Yes','Yes'])
    stargazer.add_line('Sample', ['Full','Full','Democ','Full','Full','Democ'])
    
    if name == 'ethnicity':
        stargazer.title('Panel A. Ethnicity')
        return stargazer

    else:
        stargazer.title('Panel B. Language')
        return stargazer


    
def table13_ext11(df, name, trust) :
    
    dependent = ['voice','PolStab','GovEffec','RegQual','RulLaw','ConCorr']
    
    table = [f'table{i}'
                 for i in range(6)
                ]
    
    for dep, i in zip(dependent, range(6)) :
        df_13 = df[[f'{name}_C2',f'{name}_instrument_C2_thresh',f'{name}_I',
                            'voice','PolStab','GovEffec','RegQual','RulLaw','ConCorr',
                            'trust','ethnic_party_dum','dummy_sepx_nm']].dropna(axis=0)

        y1 = df_13[f'{dep}']

        x1 = sm.add_constant(df_13[[f'{name}_C2',f'{name}_I']])
        x2 = sm.add_constant(df_13[[f'{name}_C2',f'{name}_I','trust']])
        x3 = sm.add_constant(df_13[[f'{name}_C2',f'{name}_I','trust','ethnic_party_dum','dummy_sepx_nm']])

        ins1 = sm.add_constant(df_13[[f'{name}_instrument_C2_thresh',f'{name}_I']])
        ins2 = sm.add_constant(df_13[[f'{name}_instrument_C2_thresh',f'{name}_I','trust']])
        ins3 = sm.add_constant(df_13[[f'{name}_instrument_C2_thresh',f'{name}_I','trust','ethnic_party_dum',
                                              'dummy_sepx_nm']])                               
        
        est = [f'est{i}'
               for i in range(6)
               ]
        
        est[0] = sm.OLS(y1,x1).fit(cov_type = 'HC1')
        est[1] = sm.OLS(y1,x2).fit(cov_type = 'HC1')
        est[2] = sm.OLS(y1,x3).fit(cov_type = 'HC1')
        est[3] = IV2SLS(y1, x1, ins1).fit()
        est[4] = IV2SLS(y1, x2, ins2).fit()
        est[5] = IV2SLS(y1, x3, ins3).fit()
        
        if trust == 'trust':
        
            table[i] = pd.DataFrame({'OLS / Trust' : [est[1].params.values[3],est[1].bse.values[3],est[1].pvalues[3]],
                                     'OLS / All' : [est[2].params.values[3],est[2].bse.values[3],est[2].pvalues[3]],
                                     '2SLS / Trust' : [est[4].params.values[3],est[4].bse.values[3],est[4].pvalues[3]],
                                     '2SLS / All' : [est[5].params.values[3],est[5].bse.values[3],est[5].pvalues[3]]},
                                     index = ['Trust','Standard Error','p-value'])
            table[i].index = pd.MultiIndex.from_product([[f'{dep}'], table[i].index])
            
        else:
            table[i] = pd.DataFrame({'OLS / None' : [est[0].params.values[1],est[0].bse.values[1],est[0].pvalues[1]],
                                     'OLS / Trust' : [est[1].params.values[1],est[1].bse.values[1],est[1].pvalues[1]],
                                     'OLS / All' : [est[2].params.values[1],est[2].bse.values[1],est[2].pvalues[1]],
                                     '2SLS / None' : [est[3].params.values[1],est[3].bse.values[1],est[3].pvalues[1]],
                                     '2SLS / Trust' : [est[4].params.values[1],est[4].bse.values[1],est[4].pvalues[1]],
                                     '2SLS / All' : [est[5].params.values[1],est[5].bse.values[1],est[5].pvalues[1]]},
                                     index = ['Segregation','Standard Error','p-value'])
            table[i].index = pd.MultiIndex.from_product([[f'{dep}'], table[i].index])
            
    
    table = pd.concat(table)
    
    table = table.rename(index = {'voice': 'Voice',
                                  'PolStab' : 'Political stability',
                                  'GovEffec' : 'Govern-t effectiv.',
                                  'RegQual' : 'Regul. quality',
                                  'RulLaw' : 'Rule of law',
                                  'ConCorr' : 'Control of corr'
                                  })
    table.index.names = ['Dependent Var','']
    
    
    return table

def vif_cal(input_data, dependent_col, endo, instrument, reg):
    
    if reg == '2SLS' :
        x_vars=input_data.drop([dependent_col, instrument], axis=1)
        ins_vars=input_data.drop([dependent_col, endo], axis=1)
    else :
        x_vars=input_data.drop([dependent_col, instrument], axis=1)
        ins_vars=input_data.drop([dependent_col, instrument], axis=1)
    xvar_names=x_vars.columns
    vif=list()
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]]
        x=x_vars[xvar_names.drop(xvar_names[i])]
        if reg == '2SLS' :
            rsq=IV2SLS(y,x,ins_vars).fit().rsquared
        else :
            rsq=smf.ols(formula="y~x", data=x_vars).fit().rsquared
        vif.append(round(1/(1-rsq),2))
        
    if reg == 'OLS' :
        return pd.DataFrame({'Var' : xvar_names[i], 'VIF/OLS' : vif[i]} for i in range(0,xvar_names.shape[0]))
    elif reg == '2SLS' :
        return pd.DataFrame({'Var' : xvar_names[i], 'VIF/2SLS' : vif[i]} for i in range(0,xvar_names.shape[0]))