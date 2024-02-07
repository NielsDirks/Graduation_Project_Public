import tkinter as tk
from tkinter import messagebox
from tkcalendar import DateEntry

from datetime import datetime

import os
import numpy as np


def _check_numeric_entries(*entries):
    for entry in entries:
        if not entry.get().strip():  # Check if entry is empty or contains only whitespace
            return False
        try:
            float(entry.get())
        except ValueError:
            return False  # Entry contains non-numeric value
    return True


def _check_text_entries(*entries):
    for entry in entries:
        if not entry.get().strip():  # Check if entry is empty or contains only whitespace
            return False
    return True


def _check_mutation_entry(*entries):
    for entry in entries:
        entry_text = entry.get().strip()
        if not entry_text:  # Check if entry is empty or contains only whitespace
            return False

        if '(' in entry_text:   # Check if entry is a tuple
            try:
                tuple_values = eval(entry_text)
                if not isinstance(tuple_values, tuple) or len(tuple_values) != 2:
                    return False  # Entry is not a valid tuple
                try:
                    [float(val) for val in tuple_values]
                except ValueError:
                    return False    # Entries in tuple are not of type float or int
            except (SyntaxError, ValueError, TypeError, NameError):
                return False  # Entry is not a valid tuple or contains non-numeric values
        else:   # Check if entry is a single float
            try:
                float(entry_text)
            except ValueError:
                return False  # Entry contains non-numeric value
    return True


def _validate_grid(grid):
    return os.path.exists(f'MV_Grids/{grid}')


def _validate_float(num, condition, upper_limit=None):
    try:
        if upper_limit:
            if not np.all(np.array(num) <= upper_limit):
                return False
        if condition == '>0':
            return np.all(np.array(num) > 0)
        elif condition == '>=0':
            return np.all(np.array(num) >= 0)
        elif condition == '<0':
            return np.all(np.array(num) < 0)
        elif condition == '<=0':
            return np.all(np.array(num) <= 0)
        else:
            raise ValueError("The condition that was given is not available.")
    except ValueError:
        return False


def _validate_date(date):
    try:
        datetime.strptime(date, "%d-%m-%Y")
        return True
    except ValueError:
        return False


def _validate_dates(start_date, end_date):
    try:
        start_date = datetime.strptime(start_date, "%d-%m-%Y")
        end_date = datetime.strptime(end_date, "%d-%m-%Y")
        return end_date >= start_date
    except ValueError:  # one or both dates are not in the right format, let the user fix this first
        return True


def _validate_int(num, condition):
    # check if the entered number is of type int
    try:
        num = int(num)
        if condition == '>0':
            return np.all(np.array(num) > 0)
        elif condition == '>=0':
            return np.all(np.array(num) >= 0)
        elif condition == '<0':
            return np.all(np.array(num) < 0)
        elif condition == '<=0':
            return np.all(np.array(num) <= 0)
        else:
            raise ValueError("The condition that was given is not available.")
    except ValueError:
        return False


def _validate_dso_generator(limit, step_size):
    tolerance = 1e-10   # for precision issues
    return limit / step_size % 1 < tolerance


class OptimisationSettings:
    def __init__(self):
        self.mv_grid = ''
        self.import_limit = ''
        self.export_limit = ''
        self.low_load_reactive_export = False
        self.start_date = datetime.today().strftime('%d-%m-%Y')
        self.end_date = datetime.today().strftime('%d-%m-%Y')
        self.appliance_strategy = 'dso_only'
        self.popsize = int(20)
        self.tolerance = float(0.0001)
        self.dso_gen_lim = ''
        self.dso_gen_step_size = ''
        self.de_strategy = 'best1bin'
        self.maxiter = 1000
        self.mutation_const = (0.5, 1)
        self.crossover_const = 0.7
        self.disp_progress = False
        self.pop_init_strategy = 'latinhypercube'

        self._load_last_settings()

        self.root = tk.Tk()
        self.root.title("Optimisation Settings")

        # Disable the close button on the window
        self.root.protocol("WM_DELETE_WINDOW", self._disable_close)

        # first column of optimisation settings:
        mv_grid_label = tk.Label(self.root, text="Medium voltage grid name")
        mv_grid_label.grid(row=0, column=0, padx=10, pady=10)
        mv_grid_var = tk.StringVar(value=str(self.mv_grid))
        self.mv_grid_entry = tk.Entry(self.root, textvariable=mv_grid_var)
        self.mv_grid_entry.grid(row=0, column=1, padx=10, pady=10)

        import_limit_label = tk.Label(self.root, text="Substation maximum import limit")
        import_limit_label.grid(row=1, column=0, padx=10, pady=10)
        import_limit_var = tk.StringVar(value=str(self.import_limit))
        self.import_limit_entry = tk.Entry(self.root, textvariable=import_limit_var)
        self.import_limit_entry.grid(row=1, column=1, padx=10, pady=10)
        import_unit_label = tk.Label(self.root, text="MW")
        import_unit_label.grid(row=1, column=2, padx=5, pady=10)

        export_limit_label = tk.Label(self.root, text="Substation maximum export limit")
        export_limit_label.grid(row=2, column=0, padx=10, pady=10)
        export_limit_var = tk.StringVar(value=str(self.export_limit))
        self.export_limit_entry = tk.Entry(self.root, textvariable=export_limit_var)
        self.export_limit_entry.grid(row=2, column=1, padx=10, pady=10)
        export_unit_label = tk.Label(self.root, text="MW")
        export_unit_label.grid(row=2, column=2, padx=5, pady=10)

        low_load_reactive_export_label = tk.Label(self.root, text="Allow reactive power export at low load")
        low_load_reactive_export_label.grid(row=3, column=0, padx=10, pady=10)
        self.allow_reactive_export_low_load_bool = tk.BooleanVar(value=self.low_load_reactive_export)
        allow_reactive_export_low_load_checkbox = tk.Checkbutton(self.root,
                                                                 variable=self.allow_reactive_export_low_load_bool)
        allow_reactive_export_low_load_checkbox.grid(row=3, column=1, pady=20)

        self.start_date_label = tk.Label(self.root, text="Start date")
        self.start_date_label.grid(row=4, column=0, padx=10, pady=10)
        self.start_date_entry = DateEntry(self.root, date_pattern='dd-mm-yyyy')
        self.start_date_entry.grid(row=4, column=1, padx=10, pady=10)
        self.start_date_entry.set_date(datetime.strptime(self.start_date, "%d-%m-%Y").date())

        self.end_date_label = tk.Label(self.root, text="End date")
        self.end_date_label.grid(row=5, column=0, padx=10, pady=10)
        self.end_date_entry = DateEntry(self.root, date_pattern='dd-mm-yyyy')
        self.end_date_entry.grid(row=5, column=1, padx=10, pady=10)
        self.end_date_entry.set_date(datetime.strptime(self.end_date, "%d-%m-%Y").date())

        appliance_strategy_label = tk.Label(self.root, text="Appliances that control reactive power")
        appliance_strategy_label.grid(row=6, column=0, padx=10, pady=10)
        self.appliance_strategy_var = tk.StringVar(value=self.appliance_strategy)
        appliance_strategy_options = ["dso_only", "customers_only", "both"]
        appliance_strategy_entry = tk.OptionMenu(self.root, self.appliance_strategy_var,
                                                 *appliance_strategy_options)
        appliance_strategy_entry.grid(row=6, column=1, padx=10, pady=10)

        popsize_label = tk.Label(self.root, text="Population size multiplier")
        popsize_label.grid(row=7, column=0, padx=10, pady=10)
        popsize_var = tk.StringVar(value=str(self.popsize))
        self.popsize_entry = tk.Entry(self.root, textvariable=popsize_var)
        self.popsize_entry.grid(row=7, column=1, padx=10, pady=10)

        tolerance_label = tk.Label(self.root, text="Convergene tolerance")
        tolerance_label.grid(row=8, column=0, padx=10, pady=10)
        tolerance_var = tk.StringVar(value=str(self.tolerance))
        self.tolerance_entry = tk.Entry(self.root, textvariable=tolerance_var)
        self.tolerance_entry.grid(row=8, column=1, padx=10, pady=10)

        # draw a black line to separate the two input columns
        canvas = tk.Canvas(self.root, height=400, width=2, bg="black")
        canvas.grid(row=0, column=3, rowspan=9, padx=5)

        # second column of optimisation settings:
        dso_gen_lim_label = tk.Label(self.root, text="DSO generator limit")
        dso_gen_lim_label.grid(row=0, column=4, padx=10, pady=10)
        dso_gen_lim_var = tk.StringVar(value=str(self.dso_gen_lim))
        self.dso_gen_lim_entry = tk.Entry(self.root, textvariable=dso_gen_lim_var)
        self.dso_gen_lim_entry.grid(row=0, column=5, padx=10, pady=10)
        dso_gen_lim_unit_label = tk.Label(self.root, text="Mvar")
        dso_gen_lim_unit_label.grid(row=0, column=6, padx=5, pady=10)

        dso_gen_step_size_label = tk.Label(self.root, text="DSO generator step size")
        dso_gen_step_size_label.grid(row=1, column=4, padx=10, pady=10)
        dso_gen_step_size_var = tk.StringVar(value=str(self.dso_gen_step_size))
        self.dso_gen_step_size_entry = tk.Entry(self.root, textvariable=dso_gen_step_size_var)
        self.dso_gen_step_size_entry.grid(row=1, column=5, padx=10, pady=10)
        dso_gen_step_size_unit_label = tk.Label(self.root, text="Mvar")
        dso_gen_step_size_unit_label.grid(row=1, column=6, padx=5, pady=10)

        de_strategy_label = tk.Label(self.root, text="Differential evolution strategy")
        de_strategy_label.grid(row=2, column=4, padx=10, pady=10)
        self.de_strategy_var = tk.StringVar(value=self.de_strategy)
        de_strategy_options = ["best1bin", "rand1bin", "randtobest1bin", "currenttobest1bin", "best2bin", "rand2bin",
                               "best1exp", "rand1exp", "randtobest1exp", "currenttobest1exp", "best2exp", "rand2exp"]
        de_strategy_entry = tk.OptionMenu(self.root, self.de_strategy_var, *de_strategy_options)
        de_strategy_entry.grid(row=2, column=5, padx=10, pady=10)

        maxiter_label = tk.Label(self.root, text="Maximum number of iterations")
        maxiter_label.grid(row=3, column=4, padx=10, pady=10)
        maxiter_var = tk.StringVar(value=str(self.maxiter))
        self.maxiter_entry = tk.Entry(self.root, textvariable=maxiter_var)
        self.maxiter_entry.grid(row=3, column=5, padx=10, pady=10)

        mutation_const_label = tk.Label(self.root, text="Mutation constant")
        mutation_const_label.grid(row=4, column=4, padx=10, pady=10)
        mutation_const_var = tk.StringVar(value=str(self.mutation_const))
        self.mutation_const_entry = tk.Entry(self.root, textvariable=mutation_const_var)
        self.mutation_const_entry.grid(row=4, column=5, padx=10, pady=10)

        crossover_const_label = tk.Label(self.root, text="Crossover constant")
        crossover_const_label.grid(row=5, column=4, padx=10, pady=10)
        crossover_const_var = tk.StringVar(value=str(self.crossover_const))
        self.crossover_const_entry = tk.Entry(self.root, textvariable=crossover_const_var)
        self.crossover_const_entry.grid(row=5, column=5, padx=10, pady=10)

        disp_progress_label = tk.Label(self.root, text="Display progress")
        disp_progress_label.grid(row=6, column=4, padx=10, pady=10)
        self.disp_progress_bool = tk.BooleanVar(value=self.disp_progress)
        disp_progress_checkbox = tk.Checkbutton(self.root, variable=self.disp_progress_bool)
        disp_progress_checkbox.grid(row=6, column=5, pady=20)

        pop_init_strategy_label = tk.Label(self.root, text="Population initialisation method")
        pop_init_strategy_label.grid(row=7, column=4, padx=10, pady=10)
        self.pop_init_strategy_var = tk.StringVar(value=self.pop_init_strategy)
        pop_init_strategy_options = ["latinhypercube", "random"]
        pop_init_strategy_entry = tk.OptionMenu(self.root, self.pop_init_strategy_var,
                                                *pop_init_strategy_options)
        pop_init_strategy_entry.grid(row=7, column=5, padx=10, pady=10)

        # place ok button at the bottom in the middle
        self.ok_button = tk.Button(self.root, text="OK", command=self._validate_and_destroy, state=tk.DISABLED,
                                   width=10)
        self.ok_button.grid(row=9, column=0, columnspan=7, pady=20)

        mv_grid_var.trace_variable("w", lambda *args: self._on_entry_change(self.mv_grid_entry))
        import_limit_var.trace_variable("w", lambda *args: self._on_entry_change(self.import_limit_entry))
        export_limit_var.trace_variable("w", lambda *args: self._on_entry_change(self.export_limit_entry))
        self.start_date_entry.bind("<<DateEntrySelected>>", lambda event: self._on_date_change())
        self.end_date_entry.bind("<<DateEntrySelected>>", lambda event: self._on_date_change())
        popsize_var.trace_variable("w", lambda *args: self._on_entry_change(self.popsize_entry))
        tolerance_var.trace_variable("w", lambda *args: self._on_entry_change(self.tolerance_entry))
        dso_gen_lim_var.trace_variable("w", lambda *args: self._on_entry_change(self.dso_gen_lim_entry))
        dso_gen_step_size_var.trace_variable("w",
                                             lambda *args: self._on_entry_change(self.dso_gen_step_size_entry))
        maxiter_var.trace_variable("w", lambda *args: self._on_entry_change(self.maxiter_entry))
        mutation_const_var.trace_variable("w", lambda *args: self._on_entry_change(self.mutation_const_entry))
        crossover_const_var.trace_variable("w", lambda *args: self._on_entry_change(self.crossover_const_entry))

        # shortcut to check if al values have been filled already
        self._on_entry_change(self.mv_grid_entry)

        # Bind the <Return> key event to the same function as the button click
        self.root.bind("<Return>", self._handle_return)

        # Update the geometry after widgets are placed
        self.root.update_idletasks()

        # Show the window and give it focus
        self.root.focus_force()

        # Center the window on the screen
        x_position = (self.root.winfo_screenwidth() - self.root.winfo_reqwidth()) // 2
        y_position = (self.root.winfo_screenheight() - self.root.winfo_reqheight()) // 2

        self.root.geometry(f"+{x_position}+{y_position}")

        self.root.mainloop()

    def _load_last_settings(self):
        file_name = 'last_settings.txt'
        try:
            # Read settings from file
            with open(file_name, 'r') as file:
                for line in file:
                    key, value = line.strip().split(':')
                    key = key.strip()
                    value = value.strip()

                    if key == 'mv_grid':
                        self.mv_grid = value
                    elif key == 'import_limit':
                        self.import_limit = value
                    elif key == 'export_limit':
                        self.export_limit = value
                    elif key == 'low_load_reactive_export':
                        self.low_load_reactive_export = value.lower() == 'true'
                    elif key == 'start_date':
                        self.start_date = value
                    elif key == 'end_date':
                        self.end_date = value
                    elif key == 'appliance_strategy':
                        self.appliance_strategy = value
                    elif key == 'popsize':
                        self.popsize = value
                    elif key == 'tolerance':
                        self.tolerance = value
                    elif key == 'dso_generator_limit':
                        self.dso_gen_lim = value
                    elif key == 'dso_generator_step_size':
                        self.dso_gen_step_size = value
                    elif key == 'de_strategy':
                        self.de_strategy = value
                    elif key == 'maxiter':
                        self.maxiter = value
                    elif key == 'mutation':
                        self.mutation_const = value
                    elif key == 'crossover':
                        self.crossover_const = value
                    elif key == 'disp':
                        self.disp_progress = value
                    elif key == 'init':
                        self.pop_init_strategy = value
        except FileNotFoundError:
            pass

    def _on_entry_change(self, entry):
        entry.config(bg='white')
        self.ok_button["state"] = tk.NORMAL if (_check_numeric_entries(self.import_limit_entry,
                                                                       self.export_limit_entry, self.popsize_entry,
                                                                       self.tolerance_entry, self.dso_gen_lim_entry,
                                                                       self.dso_gen_step_size_entry, self.maxiter_entry,
                                                                       self.crossover_const_entry)
                                                and (_check_text_entries(self.mv_grid_entry))
                                                and (_check_mutation_entry(self.mutation_const_entry))) else tk.DISABLED

    def _on_date_change(self):
        # Handle the date change event
        self.start_date_label.config(bg=self.root.cget('bg'))
        self.end_date_label.config(bg=self.root.cget('bg'))

    # noinspection PyMethodMayBeStatic
    def _disable_close(self):
        messagebox.showwarning("Warning", "Please enter all values and press 'OK' to continue.")

    def _handle_return(self, _):
        if self.ok_button["state"] == tk.NORMAL:
            self._validate_and_destroy()

    def _validate_entries(self):
        mv_grid_valid = _validate_grid(self.mv_grid_entry.get())
        import_limit_valid = _validate_float(float(self.import_limit_entry.get()), '>0')
        export_limit_valid = _validate_float(float(self.export_limit_entry.get()), '<0')
        start_date_valid = _validate_date(self.start_date_entry.get())
        end_date_valid = _validate_date(self.end_date_entry.get())
        dates_order_valid = _validate_dates(self.start_date_entry.get(), self.end_date_entry.get())
        popsize_valid = _validate_int(self.popsize_entry.get(), '>0')
        tolerance_valid = _validate_float(float(self.tolerance_entry.get()), '>0')
        dso_generator_limit_valid = _validate_float(float(self.dso_gen_lim_entry.get()), '>0')
        dso_generator_step_size_valid = _validate_float(float(self.dso_gen_step_size_entry.get()), '>0')
        dso_generator_valid = _validate_dso_generator(float(self.dso_gen_lim_entry.get()),
                                                      float(self.dso_gen_step_size_entry.get()))
        maxiter_valid = _validate_int(self.maxiter_entry.get(), '>0')
        mutation_const_valid = _validate_float(eval(self.mutation_const_entry.get()), '>=0', 2)
        crossover_const_valid = _validate_float(float(self.crossover_const_entry.get()), '>=0', 1)

        warning_message = ""
        if not mv_grid_valid:
            self.mv_grid_entry.config(bg='red')
            warning_message += "The grid name must be an existing folder name in the 'MV_Grids' folder. "
        if not import_limit_valid:
            self.import_limit_entry.config(bg='red')
            warning_message += "The import limit must be larger than zero. "
        if not export_limit_valid:
            self.export_limit_entry.config(bg='red')
            warning_message += "The export limit must be smaller than zero. "
        if not start_date_valid:
            self.start_date_label.config(bg='red')
            warning_message += "Dates must be in format 'dd-mm-yyyy'. "
        if not end_date_valid:
            self.end_date_label.config(bg='red')
            if start_date_valid:    # otherwise warning message is double
                warning_message += "Dates must be in format 'dd-mm-yyyy'. "
        if not dates_order_valid:
            self.start_date_label.config(bg='red')
            self.end_date_label.config(bg='red')
            warning_message += "The end date must be further in time or equal to the start date. "
        if not popsize_valid:
            self.popsize_entry.config(bg='red')
            warning_message += "The population size must be an integer larger than zero. "
        if not tolerance_valid:
            self.tolerance_entry.config(bg='red')
            warning_message += "The tolerance must be larger than zero. "
        if not dso_generator_limit_valid:
            self.dso_gen_lim_entry.config(bg='red')
            warning_message += "The DSO generator limit must be larger than zero. "
        if not dso_generator_step_size_valid:
            self.dso_gen_step_size_entry.config(bg='red')
            warning_message += "The DSO generator step size must be larger than zero. "
        if not dso_generator_valid:
            self.dso_gen_lim_entry.config(bg='red')
            self.dso_gen_step_size_entry.config(bg='red')
            warning_message += "The DSO generator limit must be a multiple of the DSO generator step size. "
        if not maxiter_valid:
            self.maxiter_entry.config(bg='red')
            warning_message += "The maximum number of iterations must be an integer larger than zero. "
        if not mutation_const_valid:
            self.mutation_const_entry.config(bg='red')
            warning_message += ("The mutation constant must be in the range [0, 2], or specified as a tuple(min, max) "
                                "where min < max and min, max are in the range [0, 2]. ")
        if not crossover_const_valid:
            self.crossover_const_entry.config(bg='red')
            warning_message += "The crossover constant must be in the range [0, 1]. "

        result = (mv_grid_valid and import_limit_valid and export_limit_valid and start_date_valid and end_date_valid
                  and dates_order_valid and popsize_valid and tolerance_valid and dso_generator_limit_valid and
                  dso_generator_step_size_valid and dso_generator_valid and maxiter_valid and mutation_const_valid and
                  crossover_const_valid)
        if not result:
            messagebox.showwarning("Input Error", warning_message)

        return result

    def _validate_and_destroy(self):
        if self._validate_entries():
            self.mv_grid = self.mv_grid_entry.get()
            self.import_limit = float(self.import_limit_entry.get())*10**6  # convert MW to W
            self.export_limit = float(self.export_limit_entry.get())*10**6  # convert MW to W
            self.low_load_reactive_export = self.allow_reactive_export_low_load_bool.get()  # bool
            self.start_date = self.start_date_entry.get()
            self.end_date = self.end_date_entry.get()
            self.appliance_strategy = self.appliance_strategy_var.get()     # dropdown menu
            self.popsize = int(self.popsize_entry.get())
            self.tolerance = float(self.tolerance_entry.get())
            self.dso_gen_lim = float(self.dso_gen_lim_entry.get())*10**6    # convert Mvar to var
            self.dso_gen_step_size = float(self.dso_gen_step_size_entry.get())*10**6    # convert Mvar to var
            self.de_strategy = self.de_strategy_var.get()
            self.maxiter = int(self.maxiter_entry.get())
            self.mutation_const = eval(self.mutation_const_entry.get())
            self.crossover_const = float(self.crossover_const_entry.get())
            self.disp_progress = self.disp_progress_bool.get()  # bool
            self.pop_init_strategy = self.pop_init_strategy_var.get()   # dropdown menu
            self.root.destroy()
            del (self.allow_reactive_export_low_load_bool, self.appliance_strategy_var, self.crossover_const_entry,
                 self.de_strategy_var, self.disp_progress_bool, self.dso_gen_lim_entry, self.dso_gen_step_size_entry,
                 self.end_date_entry, self.end_date_label, self.export_limit_entry, self.import_limit_entry,
                 self.maxiter_entry, self.mutation_const_entry, self.mv_grid_entry, self.ok_button,
                 self.pop_init_strategy_var, self.popsize_entry, self.root, self.start_date_entry,
                 self.start_date_label, self.tolerance_entry)

            # save settings to 'last_settings' file
            with open('last_settings.txt', 'w') as file:
                file.write(f"mv_grid: {self.mv_grid}\n")
                file.write(f"import_limit: {self.import_limit/1e6}\n")
                file.write(f"export_limit: {self.export_limit/1e6}\n")
                file.write(f"low_load_reactive_export: {str(self.low_load_reactive_export)}\n")
                file.write(f"start_date: {self.start_date}\n")
                file.write(f"end_date: {self.end_date}\n")
                file.write(f"appliance_strategy: {self.appliance_strategy}\n")
                file.write(f"popsize: {self.popsize}\n")
                file.write(f"tolerance: {self.tolerance}\n")
                file.write(f"dso_generator_limit: {self.dso_gen_lim/1e6}\n")
                file.write(f"dso_generator_step_size: {self.dso_gen_step_size/1e6}\n")
                file.write(f"de_strategy: {self.de_strategy}\n")
                file.write(f"maxiter: {self.maxiter}\n")
                file.write(f"mutation: {self.mutation_const}\n")
                file.write(f"crossover: {self.crossover_const}\n")
                file.write(f"disp: {str(self.disp_progress)}\n")
                file.write(f"init: {self.pop_init_strategy}\n")
