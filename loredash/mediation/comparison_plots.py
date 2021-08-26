from math import floor, ceil
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  
plt.rcParams['font.size'] = 7.5
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5


def setup_subplots(nsub = None, nrow = None, ncol = None, wsub = 2.0, hsub = 2.0, wspace = 0.2, hspace = 0.2, left = 0.45, right = 0.15, bot = 0.45, top = 0.45):   
    if nsub is None:
        nsub = nrow*ncol
    else:
        nrow = int(floor(nsub**0.5))
        ncol = int(ceil(nsub / float(nrow)))    
    width = wsub * ncol + (ncol-1)*wspace + left + right
    height = hsub * nrow + (nrow-1)*hspace + top + bot
    fig = plt.figure(figsize = (width, height))     
    gs = gridspec.GridSpec(nrow, ncol, bottom = bot/height,  top = 1.0-(top/height), left = left/width, right = 1.0-(right/width), wspace = wspace/wsub , hspace = hspace/hsub)    
    ax = []
    for row in range(nrow):
        for col in range(ncol):
            i = row*ncol + col
            if i < nsub:
                newax = fig.add_subplot(gs[row,col])
                ax.append(newax)    
    return [fig, ax, nrow, ncol]


def get_label(name):
    namemap = { #Variable name,   #Label
              'dn':            'DNI (W/m$^2$)',
              'beam':            'DNI (W/m$^2$)',
              'clearsky':        'Clear-sky DNI (W/m$^2$)',
              'tdry':            'Ambient T ($^{\circ}$C)',
              'wspd':            'Wind speed (m/s)',
              'sf_adjust_out':   'Field availability',
              'm_dot_rec':       'Rec. flow (kg/s)',
              'Q_thermal':       'Q$_{thermal}$ (MWt)',
              'T_rec_out':       'T$_{out}$ ($^{\circ}$C)',
              'e_ch_tes':        'TES charge state (MWht)',
              'mass_tes_hot':    'Mass in hot tank (kg)',
              'mass_tes_cold':   'Mass in cold tank (kg)',
              'T_tes_hot':       'Hot storage T ($^{\circ}$C)',
              'T_tes_cold':      'Cold storage T ($^{\circ}$C)',
              'q_pb':            'Cycle thermal input (MWt)',
              'P_cycle':         'Cycle gross output (MWe)',
             }
    return namemap[name]


#-----------------------------------------------------------------------------
def plot_solution(dispatch_soln, tech_outputs, datetime_start, datetime_end, savename = None):
    nday = int(0.5+(datetime_end-datetime_start).days)
    npts = len(tech_outputs['rec_clearsky_dni'])
    times = np.arange(npts) * 1./tech_outputs["time_steps_per_hour"]

    nrow = 9
    [fig, ax, nrow, ncol] = setup_subplots(nrow = nrow, ncol = 1, wsub = 3.5*nday, hsub = 0.75, wspace = 0, hspace = 0.3, left = 0.7, right = 0.7, bot = 0.5, top = 0.1)

    # DNI
    j = 0
    ax[j].plot(times, np.array(tech_outputs['rec_clearsky_dni'])*np.array(tech_outputs["rec_clearsky_fraction"]), lw = 0.75, color = 'steelblue', label = 'Actual')
    ax[j].plot(times, tech_outputs['rec_clearsky_dni'], lw = 0.75, color = 'maroon', label = 'Clear-sky')
    #ax[j].plot(times, cs.current_forecast_weather_data['dn'][p:p+n], '-', lw = 0.75, color = 'darkgoldenrod', label = 'Forecast')
    styles = ['-', '--']
    colors = ['grey', 'darkgoldenrod']
    # for i in range(nday):
    #     offset30 = True
    #     wfdata = util.read_weather_forecast(startime+datetime.timedelta(days = i), offset30)
    #     n = len(wfdata['dn'])
    #     tstart = 24*i + (-9 if offset30 else -8.5)    # Start time in forecast relative to 12am PST on the provided date
    #     ax[j].plot( tstart + np.repeat(np.arange(n+1),2)[3:-1], np.repeat(wfdata['dn'],2)[2:], lw = 0.75, color = colors[i%2], linestyle = styles[i%2], label = 'Forecast' if i in[0,1] else '')
    ax[j].set_ylabel('DNI (W/m$^2$)')
    ax[j].set_ylim([0, 1050])
    ax[j].legend()
    
    # Price
    #j += 1
    #ax[j].plot(times, cs.results['pricing_mult'][inds], lw = 0.75, color = 'steelblue')
    #ax[j].set_ylabel('Price ($/MWhe)')
    
    # Solar field adjustment (e.g. soiling if not using CD data, or soiling/tracking/offline if using CD data)
    j+=1
    ax[j].plot(times, cs.results['sf_adjust_out'][inds], lw = 0.75, color = 'grey', label = 'sf_adjust')
    ax[j].set_ylabel('Field\nAvailability')
    ax[j].set_ylim([0,1.02])
    
    
    # # Day-ahead schedule
    # j+=1
    # if cs.use_day_ahead_schedule:
    #     da_step = 1./cs.day_ahead_schedule_steps_per_hour
    #     da_times = np.linspace(0, 24, int(24/da_step)+1)
    #     for i in range(startday, nday):
    #         if cs.schedules[i] is not None:
    #             ax[j].plot(24*i+np.repeat(da_times,2)[1:-1], np.repeat(cs.schedules[i],2), lw = 0.75, color = 'k')
    #             ax[j].fill_between(24*i+np.repeat(da_times,2)[1:-1], np.repeat(cs.schedules[i],2), lw = 0.75, alpha = 0.15, color = 'grey')
    # ax[j].set_ylabel('Day-ahead \nschedule (MWhe)')
    # ax[j].set_ylim([0, 1.05*cs.design.P_ref])
    
    # Receiver thermal power (and target operating states from dispatch)
    j += 1
    ax[j].plot(times, cs.results['Q_thermal'][inds], lw = 0.75, color = 'steelblue', label = 'ssc')
    ax[j].plot(times, cs.results['q_startup'][inds], lw = 0.75, color = 'lightblue', label = 'ssc (startup)')
    if cs.is_optimize:
        ax[j].plot(times, cs.results['disp_receiver_power'][inds]/1000, '--', lw = 0.75, color = 'maroon', label = 'Dispatch')    
    ax[j].set_ylabel(get_label('Q_thermal'))   
    ax[j].set_ylim(0, 1.05*cs.design.Qrec)
    ax[j].legend(loc = 'lower left')
    if cs.control_cycle == 'CD_data' and max(cs.ssc_dispatch_targets.is_rec_sb_allowed_in) > 0:
        ax2 = ax[j].twinx()
        ax2.fill_between(times, cs.ssc_dispatch_targets.is_rec_sb_allowed_in, lw = 0.75, color = 'maroon', alpha = 0.3, label = 'Standby')
        ax2.set_ylabel('Target receiver \nstate')
        ax2.set_ylim([0, 1.02]) 
        ax2.legend(loc = 'lower right')   
    if cs.is_optimize:
        ax2 = ax[j].twinx()
        ax2.fill_between(times, cs.results['disp_receiver_on'][inds], lw = 0.75, color = 'grey', alpha = 0.3, label = 'On')
        ax2.fill_between(times, cs.results['disp_receiver_startup'][inds], lw = 0.75, color = 'darkgreen', alpha = 0.3, label = 'Startup')
        ax2.fill_between(times, cs.results['disp_receiver_standby'][inds], lw = 0.75, color = 'maroon', alpha = 0.3, label = 'Standby')
        ax2.set_ylabel('Target receiver \nstate')
        ax2.set_ylim([0, 1.02])
        ax2.legend(loc = 'lower right')  

    # Cycle thermal input (and target operating states from dispatch)
    j += 1
    ax[j].plot(times, cs.results['q_pb'][inds], lw = 0.75, color = 'steelblue', label = 'ssc')
    if cs.is_optimize:
        ax[j].plot(times, cs.results['disp_thermal_input_to_cycle'][inds]/1000, '--', lw = 0.75, color = 'maroon', label = 'Dispatch')
    if cs.control_cycle == 'CD_data':
        ax[j].plot(times, np.array(cs.ssc_dispatch_targets.q_pc_target_on_in)[inds], '--', lw = 0.75, color = 'maroon', label = 'CD Target')
    ax[j].set_ylabel('Cycle thermal\ninput (MWt)')  
    ax[j].set_ylim([0, 1.05*cs.design.get_cycle_thermal_rating() * cs.properties.cycle_max_frac])
    ax[j].legend(loc = 'lower left')  
    if cs.is_optimize:
        ax2 = ax[j].twinx()
        ax2.fill_between(times, cs.results['disp_cycle_on'][inds], lw = 0.75, color = 'grey', alpha = 0.3, label = 'On')
        ax2.fill_between(times, cs.results['disp_cycle_startup'][inds], lw = 0.75, color = 'darkgreen', alpha = 0.3, label = 'Startup')
        ax2.fill_between(times, cs.results['disp_cycle_standby'][inds], lw = 0.75, color = 'maroon', alpha = 0.3, label = 'Standby')
        ax2.set_ylabel('Target cycle \nstate')
        ax2.set_ylim([0, 1.02])
        ax2.legend(loc = 'lower right')    

    # Cycle electrical output
    j += 1
    ax[j].plot(times, cs.results['P_cycle'][inds], lw = 0.75, color = 'steelblue', label = 'ssc (gross)')
    ax[j].plot(times, cs.results['P_out_net'][inds], lw = 0.75, color = 'k', label = 'ssc (net)')
    if cs.control_cycle == 'CD_data':
        ax[j].plot(times, np.array(cs.CD_data_for_plotting['Gross Power [MW]'])[inds], '--', lw = 0.75, color = 'maroon', label = 'CD (gross)')
        ax[j].plot(times, np.array(cs.CD_data_for_plotting['Net Power [MW]'])[inds], '--', lw = 0.75, color = 'darkgoldenrod', label = 'CD (net)')
    if cs.is_optimize:
        ax[j].plot(times, cs.results['disp_electrical_output_from_cycle'][inds]/1000, '--', lw = 0.75, color = 'maroon', label = 'Dispatch (gross)')
        ax[j].plot(times, cs.results['disp_net_electrical_output'][inds]/1000, '--', lw = 0.75, color = 'darkgoldenrod', label = 'Dispatch (net)')
    ax[j].set_ylabel('Cycle gross\noutput (MWe)')  
    ax[j].set_ylim([0, 1.05*cs.design.P_ref])
    ax[j].legend()  
    
    # TES
    j += 1
    ax[j].plot(times, cs.results['e_ch_tes'][inds], lw = 0.75, color = 'steelblue', label = 'ssc')
    if cs.control_cycle == 'CD_data':
        ax[j].plot(times, np.array(cs.CD_data_for_plotting['E charge TES [MWht]'])[inds], '--', lw = 0.75, color = 'maroon', label = 'CD')
    if cs.is_optimize:
        ax[j].plot(times, cs.results['disp_tes_soc'][inds]/1000, '--', lw = 0.75, color = 'maroon', label = 'Dispatch')
    ax[j].set_ylabel('TES (MWht)')  
    ax[j].set_ylim([0, 1.05*cs.design.get_cycle_thermal_rating() * cs.design.tshours])
    ax[j].legend(loc = 'lower left')  
    
    # Receiver outlet T
    j += 1
    ax[j].plot(times, cs.results['T_rec_out'][inds], lw = 0.75, color = 'maroon', label = 'Receiver')
    if cs.control_cycle == 'CD_data' and cs.control_field == 'CD_data' and cs.control_receiver == 'CD_data':
        ax[j].plot(times, np.array(cs.CD_data_for_plotting['Rec avg Tout [C]'])[inds], '--', lw = 0.75, color = 'red', label = 'CD')
    ax[j].set_ylim([275, 580])
    ax[j].set_ylabel('Receiver \n T$_{out}$ ($^{\circ}$C)') 
    ax[j].legend()  

    # TES T
    j+=1 
    ax[j].plot(times, cs.results['T_tes_hot'][inds], lw = 0.75, color = 'maroon', label = 'Hot storage')
    if cs.control_cycle == 'CD_data':
        ax[j].plot(times, np.array(cs.CD_data_for_plotting['Hot Tank Temp [C]'])[inds], '--', lw = 0.75, color = 'red', label = 'CD (hot)')
    ax[j].set_ylabel('T$_{hot}$ ($^{\circ}$C)')  
    ax[j].set_ylim([475, 580])
    ax[j].legend(loc = 'lower left')
    ax2 = ax[j].twinx()
    ax2.plot(times, cs.results['T_tes_cold'][inds], lw = 0.75, color = 'steelblue', label = 'Cold storage')
    if cs.control_cycle == 'CD_data':
        ax2.plot(times, np.array(cs.CD_data_for_plotting['Cold Tank Temp [C]'])[inds], '--', lw = 0.75, color = 'blue', label = 'CD (cold)')
    ax2.set_ylabel('T$_{cold}$ ($^{\circ}$C)')  
    ax2.legend(loc = 'lower right')        
    ax2.set_ylim([280, 330])
    
    ax[j].set_xlabel('Time (hr)')
    
    ticks = np.arange(0, nday*24, 6)
    for j in range(nrow):
        ax[j].set_xlim(0, nday*24)
        ax[j].set_xticks(ticks)  
    
    if savename is not None:
        plt.savefig(savename, dpi = 300)
    else:
        plt.show()
 
    return
