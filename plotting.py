import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# noinspection PyUnresolvedReferences
import scienceplots
import pandas as pd
import numpy as np
import os

save_pq_plot_for_report = True
save_generation_plot_for_report = False
plt_pause_time = 0.01

# make matplotlib work with the old Enexis version of python (I guess)
matplotlib.use('TkAgg')
if save_generation_plot_for_report:
    plt.style.use(['science', 'ieee', 'no-latex'])  # makes figures in IEEE report style


class Plot:
    def __init__(self, de_info):
        if de_info.limits.shape[1] == 2:  # initialise 2D plot
            plt.ion()
            self.fig, self.ax = plt.subplots()
            if save_generation_plot_for_report:
                # noinspection PyUnresolvedReferences
                self.ax.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
                # noinspection PyUnresolvedReferences
                self.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            self.feasible_data, = self.ax.plot(0, 0, '.g', label='Feasible')
            self.unfeasible_data, = self.ax.plot(0, 0, '.r', label='Unfeasible')
            self.best_data, = self.ax.plot(0, 0, 'sb', markersize=5, label='Best')
            self.ax.grid()
            self.ax.set_title('Generation 0')
            self.ax.set_xlabel('Q device 1 [Mvar]')
            self.ax.set_ylabel('Q device 2 [Mvar]')
            self.ax.legend()
            scale = de_info.settings.dso_gen_step_size / 1e6
            if de_info.pgm.n_dso_gens == 1:
                self.ax.set_xlim(de_info.limits[0][0] * scale, de_info.limits[1][0] * scale)
                self.ax.set_ylim(de_info.limits[0][1] / 1e6, de_info.limits[1][1] / 1e6)
            elif de_info.pgm.n_dso_gens == 2:
                self.ax.set_xlim(de_info.limits[0][0] * scale, de_info.limits[1][0] * scale)
                self.ax.set_ylim(de_info.limits[0][1] * scale, de_info.limits[1][1] * scale)
            else:
                self.ax.set_xlim(de_info.limits[0][0] / 1e6, de_info.limits[1][0] / 1e6)
                self.ax.set_ylim(de_info.limits[0][1] / 1e6, de_info.limits[1][1] / 1e6)
            plt.tight_layout()
        elif de_info.limits.shape[1] == 3:  # initialise 3D plot
            plt.ion()
            self.fig, self.ax = plt.subplots(subplot_kw={'projection': '3d'})
            if save_generation_plot_for_report:
                # noinspection PyUnresolvedReferences
                self.ax.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
                # noinspection PyUnresolvedReferences
                self.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            self.feasible_data, = self.ax.plot(0, 0, 0, '.g', label='Feasible')
            self.unfeasible_data, = self.ax.plot(0, 0, 0, '.r', label='Unfeasible')
            self.best_data, = self.ax.plot(0, 0, 0, 'sb', markersize=5, label='Best')
            self.ax.grid()
            self.ax.set_title('Generation 0')
            self.ax.set_xlabel('Q device 1 [Mvar]')
            self.ax.set_ylabel('Q device 2 [Mvar]')
            self.ax.set_zlabel('Q device 3 [Mvar]')
            self.ax.legend()
            scale = de_info.settings.dso_gen_step_size / 1e6
            if de_info.pgm.n_dso_gens == 1:
                self.ax.set_xlim(de_info.limits[0][0] * scale, de_info.limits[1][0] * scale)
                self.ax.set_ylim(de_info.limits[0][1] / 1e6, de_info.limits[1][1] / 1e6)
                self.ax.set_zlim(de_info.limits[0][2] / 1e6, de_info.limits[1][2] / 1e6)
            elif de_info.pgm.n_dso_gens == 2:
                self.ax.set_xlim(de_info.limits[0][0] * scale, de_info.limits[1][0] * scale)
                self.ax.set_ylim(de_info.limits[0][1] * scale, de_info.limits[1][1] * scale)
                self.ax.set_zlim(de_info.limits[0][2] / 1e6, de_info.limits[1][2] / 1e6)
            elif de_info.pgm.n_dso_gens == 3:
                self.ax.set_xlim(de_info.limits[0][0] * scale, de_info.limits[1][0] * scale)
                self.ax.set_ylim(de_info.limits[0][1] * scale, de_info.limits[1][1] * scale)
                self.ax.set_zlim(de_info.limits[0][2] * scale, de_info.limits[1][2] * scale)
            else:
                self.ax.set_xlim(de_info.limits[0][0] / 1e6, de_info.limits[1][0] / 1e6)
                self.ax.set_ylim(de_info.limits[0][1] / 1e6, de_info.limits[1][1] / 1e6)
                self.ax.set_zlim(de_info.limits[0][2] / 1e6, de_info.limits[1][2] / 1e6)
            plt.tight_layout()
        else:  # initialise no plot
            self.fig = None

    def plot_generation(self, de_info, nit):
        date_time = de_info.pgm.measurement_info.date_time

        # 2D plot (two optimisation variables)
        if de_info.population.shape[1] == 2:
            data = de_info._scale_parameters(de_info.population) / 1e6  # [Mvar]
            if de_info.pgm.n_dso_gens:  # if there are DSO gens, change values from step to Mvar value
                # multiply dso gen step with dso gen step size to get Mvar value
                data[:, :de_info.pgm.n_dso_gens] = data[:, :de_info.pgm.n_dso_gens] * de_info.settings.dso_gen_step_size
            self.feasible_data.set_data(data[:, 0][de_info.feasible], data[:, 1][de_info.feasible])
            self.unfeasible_data.set_data(data[:, 0][~de_info.feasible], data[:, 1][~de_info.feasible])
            self.best_data.set_data(data[0, 0], data[0, 1])
            self.ax.set_title(f'{date_time} Generation {nit}')
            if save_generation_plot_for_report:
                self.ax.legend(frameon=True, facecolor='white', framealpha=0.75, loc='upper right')
                # save figure
                plt.savefig('Plots/Generation_0.pdf', format='pdf')
            plt.pause(plt_pause_time)

        # 3D plot (three optimisation variables)
        if de_info.population.shape[1] == 3:
            data = de_info._scale_parameters(de_info.population) / 1e6  # [Mvar]
            if de_info.pgm.n_dso_gens:  # if there are DSO gens, change values from step to Mvar value
                # multiply dso gen step with dso gen step size to get Mvar value
                data[:, :de_info.pgm.n_dso_gens] = data[:, :de_info.pgm.n_dso_gens] * de_info.settings.dso_gen_step_size
            self.feasible_data.set_data(data[:, 0][de_info.feasible], data[:, 1][de_info.feasible])
            self.feasible_data.set_3d_properties(data[de_info.feasible, 2])
            self.unfeasible_data.set_data(data[:, 0][~de_info.feasible], data[:, 1][~de_info.feasible])
            self.unfeasible_data.set_3d_properties(data[~de_info.feasible, 2])
            self.best_data.set_data(data[0, 0], data[0, 1])
            self.best_data.set_3d_properties(data[0, 2])
            self.ax.set_title(f'{date_time} Generation {nit}')
            plt.pause(plt_pause_time)

    def close_plot(self):
        plt.close(self.fig)
        plt.ioff()


def _flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


class OutputData:
    def __init__(self, optimisation_settings):
        self.date_time = []
        self.original_p_pu = []
        self.original_q_pu = []
        self.original_cost = []
        self.original_feasible = []
        self.optimised_p_pu = []
        self.optimised_q_pu = []
        self.optimised_cost = []
        self.optimised_cost_losses = []
        self.optimised_cost_dso_devices = []
        self.optimised_cost_customer_compensation = []
        self.optimised_feasible = []
        self.optimisation_variables_output = []
        self.total_function_evaluations = 0

        self.import_capacity = optimisation_settings.import_limit
        export_capacity = -optimisation_settings.export_limit
        self.max_capacity = max(self.import_capacity, export_capacity)
        self.export_pu = -export_capacity/self.import_capacity
        self.grid_name = optimisation_settings.mv_grid

    def store_data(self, optimisation_result):
        self.date_time.append(optimisation_result.date_time)
        self.original_p_pu.append(optimisation_result.original_pfc.p_pu)
        self.original_q_pu.append(optimisation_result.original_pfc.q_pu)
        self.original_cost.append(optimisation_result.original_pfc.cost)
        self.original_feasible.append(optimisation_result.original_pfc.feasible)
        self.optimised_p_pu.append(optimisation_result.optimised_pfc.p_pu)
        self.optimised_q_pu.append(optimisation_result.optimised_pfc.q_pu)
        self.optimised_cost.append(optimisation_result.optimised_pfc.cost)
        self.optimised_cost_losses.append(optimisation_result.optimised_pfc.cost_losses)
        self.optimised_cost_dso_devices.append(optimisation_result.optimised_pfc.cost_dso_devices)
        self.optimised_cost_customer_compensation.append(optimisation_result.optimised_pfc.cost_customer_compensation)
        self.optimised_feasible.append(optimisation_result.optimised_pfc.feasible)
        self.optimisation_variables_output.append(np.round(optimisation_result.x, -3)/1000)
        self.total_function_evaluations += optimisation_result.nfev

    def save_data(self, optimisation_settings):
        # Create a DataFrame
        df = pd.DataFrame({
            'Date-Time': self.date_time,
            'Original P pu': _flatten(self.original_p_pu),
            'Original Q pu': _flatten(self.original_q_pu),
            'Original cost': self.original_cost,
            'Original feasible': self.original_feasible,
            'Optimised P pu': _flatten(self.optimised_p_pu),
            'Optimised Q pu': _flatten(self.optimised_q_pu),
            'Optimised cost': self.optimised_cost,
            'Optimised cost losses': self.optimised_cost_losses,
            'Optimised cost DSO devices': self.optimised_cost_dso_devices,
            'Optimised cost customer compensation': self.optimised_cost_customer_compensation,
            'Optimised feasible': self.optimised_feasible,
            'Optimisation variables output': self.optimisation_variables_output
        })

        # make Output folder, continue if it already exists
        try:
            os.makedirs('Output')
        except FileExistsError:
            pass    # just continue saving the data

        # the user can have this file open. Only continue if the file is closed, otherwise the file is not generated
        # and all optimisation outcomes are lost
        output_path = 'Output/' + optimisation_settings.mv_grid + '.xlsx'
        while True:  # repeat until the try statement succeeds
            try:
                test = open(output_path, 'r+')
                test.close()
                break  # exit the loop
            except PermissionError:
                input(f"Could not open {output_path}! Please close this file and press Enter to retry.")
                # restart the loop
            except FileNotFoundError:
                break   # continue to produce the file

        # Save to Excel file
        df.to_excel(output_path, index=False)

    def plot_data(self):
        if save_pq_plot_for_report:
            plt.style.use(['science', 'ieee', 'no-latex'])  # makes figures in IEEE report style
            ms = 1
        else:
            ms = 5
        plt.figure()
        plt.rc('axes', axisbelow=True)  # plot axis behind figure data
        plt.axvline(color='k', zorder=1, linewidth=0.5)
        plt.axhline(color='k', zorder=1, linewidth=0.5)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.zorder = 0    # place spines (border with ticker marks) behind to be plotted boxes and borders
        plt.plot(self.original_p_pu, self.original_q_pu, '.m', zorder=3, ms=ms, mew=0.1, label='Original')   # ms=0.1
        plt.plot(self.optimised_p_pu, self.optimised_q_pu, '.c', zorder=3, ms=ms, mew=0.1, label='Optimised')    # ms=0.1
        ax.add_patch(mpatches.Rectangle((self.export_pu, -0.1), -self.export_pu + 1, 0.6,
                                        color='#b2d8b2',
                                        linewidth=0.5,
                                        zorder=0))
        ax.add_patch(mpatches.Rectangle((-0.25, -0.1), 0.5, 0.1,
                                        color='#ffd580',
                                        linewidth=0.5,
                                        zorder=0))
        ax.add_patch(mpatches.Rectangle((-0.25, -0.1), 0.5, 0.1,
                                        fill=False,
                                        color='orange',
                                        linewidth=0.5,
                                        zorder=1))
        ax.add_patch(mpatches.Rectangle((self.export_pu, -0.1), -self.export_pu + 1, 0.6,
                                        fill=False,
                                        color='r',
                                        linewidth=0.5,
                                        zorder=2))
        plt.title(f"{self.grid_name} PQ-data")
        # plt.title('Reactive Power Range and Field Measurement')    # for the image in the final report
        plt.xlabel('P [pu]')
        plt.ylabel('Q [pu]')
        plt.grid()
        plt.xlim([min([self.export_pu, min(self.original_p_pu)]), max(1, max(self.original_p_pu))])
        plt.ylim([min(-0.3, min(self.original_q_pu)), max(0.75, max(self.original_q_pu))])
        plt.legend(markerscale=5)

        if save_pq_plot_for_report:
            # make Plots folder, continue if it already exists
            try:
                os.makedirs('Plots')
            except FileExistsError:
                pass  # just continue saving the data

            output_path = "Plots/PQ_output_plot.pdf"
            # the user can have this file open. Only continue if the file is closed, otherwise the plot is not generated
            # and all optimisation outcomes are lost
            while True:  # repeat until the try statement succeeds
                try:
                    test = open(output_path, 'r+')
                    test.close()
                    break  # exit the loop
                except PermissionError:
                    input(f"Could not open {output_path}! Please close this file and press Enter to retry.")
                    # restart the loop
                except FileNotFoundError:
                    break   # continue to produce the plot

            plt.savefig(output_path, format='pdf')

        plt.show()
