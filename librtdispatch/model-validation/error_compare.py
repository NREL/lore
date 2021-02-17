import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

compare_results = ["fixed_ACC_recsim1_updated_Qcalc", "sliding_ACC_recsim1_updated_Qcalc", "UD_pc_att3_controller_fixes"]
plot_colors = ['blue', 'red', 'black']
legend_names = ['Fixed pressure', 'Sliding pressure', 'User-defined model']

# creating a color map for plotting
color_map = {}
legand_map = {}
for key, c, l in zip(compare_results, plot_colors, legend_names):
    color_map[key] = c
    legand_map[key] = l

## gather error data
error_data = {}
for res_fname in compare_results:
    error_data[res_fname] =  pickle.load(open('figures/' + res_fname +'/calc_errors.pkl', "rb"))

## Plot error data
plt_err_key_name = {'int': ['gen','avg_err_cap']}

milestones = {'gen': 0.03,
            'avg_err_cap': 0.05}
xlabels = {'gen': 'Daily Generation Error [W$_{gross}$ / W$_{gross}$]',
            'avg_err_cap': r'Normalized Average TES Error [$\bar{e}_{charge}$ / E$_{capacity}$]'}
save_names = {'gen': 'gen_err_compare',
                'avg_err_cap': 'tes_err_compare'}

save_plot = False
save_dir = 'figures/'

for err_key in plt_err_key_name.keys():
    for err_val in plt_err_key_name[err_key]:
        Nf = 16
        fig = plt.figure(figsize=(7,7))
        plt.tick_params(axis='both', labelsize = Nf)
        for case_key in error_data.keys():
            errd = error_data[case_key][err_key][err_val]
            errs = [errd[key] for key in errd.keys()]            
            n = len(errs)
            y = np.linspace(0, 1, n, endpoint = False) + 1./n
            plt.plot(np.sort(errs), y, 'o-', lw = 2.0, ms = 4.0, color = color_map[case_key], label = legand_map[case_key])
 
        plt.xlabel(xlabels[err_val], fontsize = Nf)
        plt.ylabel('Cumulative distribution', fontsize = Nf)
        plt.fill_between([-milestones[err_val], milestones[err_val]], [1.0, 1.0], [0.0, 0.0], color = '0.8', alpha = 0.3)
        plt.ylim([0,1]) 
        plt.tight_layout()
        plt.legend(fontsize=Nf-2)
        if save_plot:
            plt.savefig(save_dir + save_names[err_val] + '.png')
            fig.clf()
            plt.close()
        else:
            plt.show()