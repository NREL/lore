import numpy as np
from datetime import datetime
import datetime
import matplotlib.pyplot as plt

import functions

def cycle_validation_plot(ssc, cd, date, Nf = 14, save_plot = False, save_dir = "figures/", debug = False):
    nsteps = len(ssc['P_cycle'])
    date_split = date.split('_')
    dates = [datetime.datetime(int(date_split[0]), int(date_split[1]), int(date_split[2])) + datetime.timedelta(minutes=(24*60/nsteps)*i) for i in range(len(ssc['P_cycle']))]
    ### Plotting comparison ssc solution vs cd data
    fig = plt.figure(figsize=(20,10))
    Nrows = 2
    Ncols = 4
    subplt = 1
    # Mass to power cycle
    #postssc_mass = [q/(functions.salt_cp(0.5*(Tc+Th)+273.15)*(Th-Tc)*1.e-6) for q,Tc,Th in zip(cd['Q into cycle [MW]'], ssc['T_tes_cold'], ssc['T_tes_hot'])]
    plt.subplot(Nrows,Ncols,subplt)
    plt.tick_params(axis='both', labelsize = Nf)
    plt.plot(dates, cd['Flow into cycle [kg/s]'], 'k-', label = 'CD Data')
    plt.plot(dates, ssc['m_dot_tes_hot_out'], 'r--', label = 'SSC')
    #plt.plot(dates, ssc['m_dot_pc_to_tes_cold'], 'b--', label = 'SSC')
    #plt.plot(dates, postssc_mass, 'r:', label = 'SSC Post Calc')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.ylabel('Mass flow to cycle [kg/s]', fontsize = Nf)
    subplt += 1
    # Heat into cycle
    #postssc_heat = [mass*functions.salt_cp(0.5*(Tc+Th)+273.15)*(Th-Tc)*1.e-6 for mass,Tc,Th in zip(ssc['m_dot_tes_hot_out'], ssc['T_pc_out'], ssc['T_pc_in'])]
    plt.subplot(Nrows,Ncols,subplt)
    plt.tick_params(axis='both', labelsize = Nf)
    plt.plot(dates, cd['Q into cycle [MW]'], 'k-', label = 'Target Heat')
    plt.plot(dates, ssc['q_pb'], 'r--', label = 'SSC')
    #plt.plot(dates, postssc_heat, 'r:', label = 'SSC Post Calc')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.ylabel('Heat into cycle [MWt]', fontsize = Nf)
    subplt += 1
    # Net power
    plt.subplot(Nrows,Ncols,subplt)
    plt.tick_params(axis='both', labelsize = Nf)
    plt.plot(dates, [gross - net for gross,net in zip(cd['Gross Power [MW]'], cd['Net Power [MW]'])], 'k-', label = 'CD Data')
    plt.plot(dates, [gross - net for gross,net in zip(ssc['P_cycle'], ssc['P_out_net'])], 'r--', label = 'SSC')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.ylabel('Power Losses [MWe]', fontsize = Nf)
    subplt += 1
    # Gross power
    plt.subplot(Nrows,Ncols,subplt)
    plt.tick_params(axis='both', labelsize = Nf)
    plt.plot(dates, cd['Gross Power [MW]'], 'k-', label = 'CD Data')
    if debug:
        plt.plot(dates, cd['Net Power [MW]'], 'b:', label = 'CD Data - net')
    plt.plot(dates, ssc['P_cycle'], 'r--', label = 'SSC')
    #plt.plot(dates, ssc['P_out_net'], 'r:', label = 'SSC net')
    #plt.plot(dates, cd['Ambient Temp [F]'], 'b--', label = 'Amb. Temp')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.ylabel('Gross Power [MWe]', fontsize = Nf)
    subplt += 1
    # Tank Temperature (cold)
    #postssc_coldT = [Th - q/(mass*functions.salt_cp(0.5*(290.+Th)+273.15)*1.e-6) for q, mass,Th in zip(ssc['q_pb'], ssc['m_dot_tes_hot_out'], ssc['T_tes_hot'])]
    plt.subplot(Nrows,Ncols,subplt)
    plt.tick_params(axis='both', labelsize = Nf)
    plt.plot(dates, cd['Cold Tank Temp [C]'], 'b-', label = 'CD Data')
    plt.plot(dates, ssc['T_tes_cold'], 'b--', label = 'SSC')
    if debug:
        plt.plot(dates, cd['Cycle Return Temp [C]'], 'b:', label = 'CD Data - cycle outlet')
        plt.plot(dates, ssc['T_pc_out'], 'b-.', label = 'SSC - cycle outlet')
    #plt.plot(dates, postssc_coldT, 'b:', label = 'SSC post calc.')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.ylabel('Cold Tank Temperature [C]', fontsize = Nf)
    subplt += 1
    # Tank Temperature (hot)
    plt.subplot(Nrows,Ncols,subplt)
    plt.tick_params(axis='both', labelsize = Nf)
    plt.plot(dates, cd['Hot Tank Temp [C]'], 'r-', label = 'CD Data')
    plt.plot(dates, ssc['T_tes_hot'],  'r--', label = 'SSC')
    #plt.plot(dates, ssc['T_pc_in'], 'r:', label = 'SSC - cycle inlet')

    #plt.plot(dates, ssc['T_rec_out'],  'k--', label = 'SSC - rec Tout')
    #plt.fill_between() # would be nice
    #plt.plot(dates, [x*540. for x in ssc['is_rec_su_allowed_in']],  'k-', label = 'SSC - rec bool')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.ylabel('Hot Tank Temperature [C]', fontsize = Nf)
    subplt += 1
    # Tank Mass
    plt.subplot(Nrows,Ncols,subplt)
    plt.tick_params(axis='both', labelsize = Nf)
    plt.plot(dates, cd['Cold Tank Mass [kg]'], 'b-', label = 'CD Data')
    plt.plot(dates, ssc['mass_tes_cold'], 'b--', label = 'SSC')
    plt.plot(dates, cd['Hot Tank Mass [kg]'], 'r-', label = 'CD Data')
    plt.plot(dates, ssc['mass_tes_hot'], 'r--', label = 'SSC')
    # Total mass for debugging 
    if debug:
        plt.plot(dates, [hot + cold for hot,cold in zip(cd['Hot Tank Mass [kg]'],cd['Cold Tank Mass [kg]'])], 'k-', label = 'CD Data')
        plt.plot(dates, [hot + cold for hot,cold in zip(ssc['mass_tes_hot'],ssc['mass_tes_cold'])], 'k--', label = 'SSC')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.ylabel('Salt Mass in Tank [kg]', fontsize = Nf)
    subplt += 1
    # Gross Power, TES hot, and cold mass Error
    plt.subplot(Nrows,Ncols,subplt)
    plt.tick_params(axis='both', labelsize = Nf)
    #plt.plot(dates, [100*(x - y)/max(y,1) for x,y in zip(ssc['P_cycle'], cd['Gross Power [MW]'])], 'k-', label = 'Gross Power')
    #plt.plot(dates, [100*(x - y)/max(y,1) for x,y in zip(ssc['e_ch_tes'], cd['E charge TES [MWht]'])], 'r--', label = 'TES Charge State')

    plt.plot(dates, cd['E charge TES [MWht]'], 'r-', label = 'CD Data')
    plt.plot(dates, ssc['e_ch_tes'], 'r--', label = 'SSC')
    #plt.ylim([-20, 20])
    plt.legend()
    plt.gcf().autofmt_xdate()
    #plt.ylabel('Instantous Relative Error [%]', fontsize = Nf)
    plt.ylabel('TES Charge State [MWh]', fontsize = Nf)
    subplt += 1

    plt.tight_layout()
    if save_plot:
        plt.savefig(save_dir + date + '.png')
        fig.clf()
        plt.close()
    else:
        plt.show()

def error_plot(errdata, ylabel = '', Nf = 16, save_plot = False, savename = 'error', save_dir = "figures/"):
    errd = dict(errdata)
    errd.update((x, y*100.) for x, y in errd.items())
    fig = plt.figure(figsize=(12.5,5))
    plt.tick_params(axis='both', labelsize = Nf)
    plt.bar(errd.keys(), errd.values())
    plt.ylabel(ylabel, fontsize = Nf)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_plot:
        plt.savefig(save_dir + savename + '.png')
        fig.clf()
        plt.close()
    else:
        plt.show()

    
def cum_dist_error_plot(errd, milestone = 0.05, xlabel = '', Nf = 16, save_plot = False, savename = 'error_dist', save_dir = "figures/"):
    errs = [errd[key] for key in errd.keys()]
    n = len(errs)
    y = np.linspace(0, 1, n, endpoint = False) + 1./n
    fig = plt.figure(figsize=(7,7))
    plt.tick_params(axis='both', labelsize = Nf)
    plt.plot(np.sort(errs), y, 'o-', lw = 2.0, ms = 4.0, color = 'steelblue')
    plt.xlabel(xlabel, fontsize = Nf)
    plt.ylabel('Cumulative distribution', fontsize = Nf)
    plt.fill_between([-milestone, milestone], [1.0, 1.0], [0.0, 0.0], color = '0.8', alpha = 0.3)
    plt.ylim([0,1]) 
    plt.tight_layout  
    if save_plot:
        plt.savefig(save_dir + savename + '.png')
        fig.clf()
        plt.close()
    else:
        plt.show()