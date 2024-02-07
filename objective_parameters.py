import numpy as np
import warnings
import power_grid_model.errors

time_scale = 0.25                       # [h]
inductor_bank_use_time = 20/100*35040   # [quarters/year]
price_inductor_bank = 200000            # [€]
payback_period = 20                     # [years]
price_ms_field = 200000                 # [€] price for a field at the main station
price_tds_field = 100000                # [€] price for a field at a transport distribution station
# price_ms_dso_bank and price_tds_dso_bank in [€/quarter]
price_ms_dso_bank = round((price_inductor_bank + price_ms_field) / (payback_period * inductor_bank_use_time), 2)
price_tds_dso_bank = round((price_inductor_bank + price_tds_field) / (payback_period * inductor_bank_use_time), 2)

price_customers = [0.2, 0.2, 0.2]
price_active_power = 100                # [€/MWh]


# set user warning format
def _one_line_user_warning(message, category):  # , filename, lineno, file=None, line=None):
    return '%s: %s\n' % (category.__name__, message)
    # return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


warnings.formatwarning = _one_line_user_warning


class ObjectiveParameters:
    def __init__(self, converged, de_info=None, pgm_output_data=None):
        if converged:
            optimisation_settings = de_info.settings
            pgm_model = de_info.pgm

            # ToDo: assign gen ids to the right array from given Vision export, not hard coded as below
            # hard coded 'Hoofdstation' and 'Transportverdeelstation' pgm node ids:
            pgm_model.main_station_gen_ids = [29]
            pgm_model.transport_distribution_station_gen_ids = [30]

            try:
                self.line_losses = np.sum(pgm_output_data['line']['p_from'] + pgm_output_data['line']['p_to'])
            except KeyError:    # if no cables are found
                self.line_losses = 0

            try:
                self.transformer_losses = np.sum(pgm_output_data['transformer']['p_from'] +
                                                 pgm_output_data['transformer']['p_to'])
            except KeyError:    # if no transformers are found
                self.transformer_losses = 0

            self.total_losses = self.line_losses + self.transformer_losses  # [W]
            self.price_losses = self.total_losses / 1e6 * time_scale * price_active_power   # [€]

            customer_count = 0
            self.price_customer_gens = 0
            self.price_dso_gens = 0
            for pgm_gen_id in pgm_model.rpc_gen_ids:
                # find the output data of the gen with id 'pgm_gen_id'
                gen_output = pgm_output_data['sym_gen'][pgm_output_data['sym_gen']['id'] == pgm_gen_id]
                if pgm_gen_id in pgm_model.gen_ids_orig:   # generator of a customer
                    # first get original setting (from assigning measurement data)
                    # Note if q is positive --> capacitive, negative --> inductive for generators
                    q_orig = pgm_model.settings_gens_orig['q_specified'][pgm_model.settings_gens_orig['id'] ==
                                                                         pgm_gen_id]
                    # calculate the total price with the price this customer asks
                    # assume inductive and capacitive reactive power is equal in value (abs())
                    self.price_customer_gens += float(abs(gen_output['q'] - q_orig) / 1e6 * time_scale *
                                                      price_customers[customer_count])                      # [€]
                    customer_count += 1
                else:                               # generator of DSO
                    # assume inductive and capacitive reactive power is equal in value (abs())
                    # if a DSO bank (cap or ind) has to be place (q of gen_output is not 0), add the price of 1 bank
                    if gen_output['q']:
                        if pgm_gen_id in pgm_model.main_station_gen_ids:    # add price for bank at main station
                            self.price_dso_gens += price_ms_dso_bank    # [€]
                        elif pgm_gen_id in pgm_model.transport_distribution_station_gen_ids:    # price at TD station
                            self.price_dso_gens += price_tds_dso_bank   # [€]
                        else:
                            raise ValueError("Not yet implemented.")

            self.total_price = self.price_losses + self.price_customer_gens + self.price_dso_gens   # [€]

            self.value = self.total_price

            # check constraints
            # first constraint: reactive power exchange at slack bus (TSO-DSO exchange), see article 9.15 of grid code
            self._q_exchange_constraint(pgm_output_data, optimisation_settings)
            # second constraint: voltage limits at nodes
            # set limits
            upper_voltage_limit = 22/20     # 1.1 pu,       11/  22 kV
            lower_voltage_limit = 19.5/20   # 0.975 pu      9.75/19.5 kV
            self._voltage_constraint(pgm_output_data, upper_voltage_limit, lower_voltage_limit)
            # third constraint: component capacity limits
            self._component_capacity_constraint(pgm_output_data, pgm_model)

            self.feasible = self.q_exchange_feasible and self.voltages_feasible and self.capacity_feasible
            self.constraint_violation = [self.q_exchange_excess, self.voltages_excess, self.capacity_excess]
        else:
            self.value = np.inf
            self.feasible = False
            self.constraint_violation = [np.inf, np.inf, np.inf]

    def _q_exchange_constraint(self, pgm_output_data, optimisation_settings):
        # get (re)active power exchange values. Positive: from TSO to DSO, negative: from DSO to TSO
        p_exchange = float(pgm_output_data['source']['p'])  # W
        q_exchange = float(pgm_output_data['source']['q'])  # var
        # get the pu value for (re)active power
        p_base = optimisation_settings.import_limit
        q_base = max(optimisation_settings.import_limit, -optimisation_settings.export_limit)
        p_pu = p_exchange / p_base
        q_pu = q_exchange / q_base
        if q_pu > 0.48:  # more reactive power is imported by DSO than allowed
            self.q_exchange_feasible = False
            self.q_exchange_excess = q_exchange - 0.48 * q_base
        elif optimisation_settings.low_load_reactive_export:  # Q export limit is 10% for all P
            if q_pu < -0.1:  # negative because of export
                self.q_exchange_feasible = False
                self.q_exchange_excess = abs(q_exchange) - 0.1 * q_base
            else:
                self.q_exchange_feasible = True
                self.q_exchange_excess = 0
        else:  # Q export limit is 0% for -25% <= P <= 25% and 10% for all other P
            if -0.25 <= p_pu <= 0.25:
                if q_pu < 0:
                    self.q_exchange_feasible = False
                    self.q_exchange_excess = abs(q_exchange)
                else:
                    self.q_exchange_feasible = True
                    self.q_exchange_excess = 0
            else:  # p_pu > 0.25 || p_pu < -0.25
                if q_pu < -0.1:
                    self.q_exchange_feasible = False
                    self.q_exchange_excess = abs(q_exchange) - 0.1 * q_base
                else:
                    self.q_exchange_feasible = True
                    self.q_exchange_excess = 0

    def _voltage_constraint(self, pgm_output_data, upper_voltage_limit, lower_voltage_limit):
        # get all pu voltages and voltages of every node
        voltages = pgm_output_data['node']['u']
        voltages_pu = pgm_output_data['node']['u_pu']
        non_zero_idx = voltages != 0
        if not np.all(non_zero_idx):  # remove nodes that are not energised, but give the user a warning
            warnings.warn("One or more nodes are not energised.", UserWarning)
            voltages = voltages[non_zero_idx]
            voltages_pu = voltages_pu[non_zero_idx]
        nom_voltages = voltages / voltages_pu
        # now check the voltages, all voltages should be between the limits
        voltage_too_high = voltages_pu >= upper_voltage_limit
        voltage_too_low = voltages_pu <= lower_voltage_limit

        # ignore low voltage node voltage violations, give the user a warning if lv nodes are present
        lv_node_idx = np.where(nom_voltages < 1000)[0]
        nom_voltages = np.delete(nom_voltages, lv_node_idx)
        voltages = np.delete(voltages, lv_node_idx)
        voltages_pu = np.delete(voltages_pu, lv_node_idx)
        voltage_too_high = np.delete(voltage_too_high, lv_node_idx)
        voltage_too_low = np.delete(voltage_too_low, lv_node_idx)
        if lv_node_idx:
            warnings.warn("Voltage violations on low voltage nodes are ignored.", UserWarning)

        if all(lower_voltage_limit <= v <= upper_voltage_limit for v in voltages_pu):  # no voltage violation
            self.voltages_feasible = True
            self.voltages_excess = 0
        else:  # one or more node voltages are too high or too low
            self.voltages_feasible = False
            upper_excess = np.sum(voltages[voltage_too_high] - nom_voltages[voltage_too_high] * upper_voltage_limit)
            lower_excess = np.sum(nom_voltages[voltage_too_low] * lower_voltage_limit - voltages[voltage_too_low])
            self.voltages_excess = upper_excess + lower_excess

    def _component_capacity_constraint(self, pgm_output_data, pgm_model):
        # find overloaded cables, transformers and generators
        # cables
        cable_loadings = pgm_output_data['line']['loading']
        cable_apparent_powers = np.maximum(abs(pgm_output_data['line']['s_from']),
                                           abs(pgm_output_data['line']['s_to']))
        overloaded_cables = cable_loadings >= 1     # ndarray with True or False for every cable

        # transformers
        transformer_loadings = pgm_output_data['transformer']['loading']
        transformer_apparent_powers = np.maximum(abs(pgm_output_data['transformer']['s_from']),
                                                 abs(pgm_output_data['transformer']['s_to']))
        overloaded_transformers = transformer_loadings >= 1     # ndarray with True or False for every transformer

        # generators
        gens_to_check = pgm_model.gen_ids_orig
        gens_to_check_result = pgm_output_data['sym_gen'][np.isin(pgm_output_data['sym_gen']['id'], gens_to_check)]
        gens_to_check_s = gens_to_check_result['s']     # apparent power of the to be checked generators
        gens_limits = pgm_model.generators_capacity_limit   # [VA]
        overloaded_generators = np.array(gens_to_check_s > gens_limits)     # ndarray with True or False for every gen

        # determine feasibility and constraint violation
        if overloaded_cables.any() or overloaded_transformers.any() or overloaded_generators.any():
            # if any component is overloaded, do following
            self.capacity_feasible = False
            cable_capacity_excess = np.sum((cable_loadings[overloaded_cables] - 1) *
                                           cable_apparent_powers[overloaded_cables])
            transformer_capacity_excess = np.sum((transformer_loadings[overloaded_transformers] - 1) *
                                                 transformer_apparent_powers[overloaded_transformers])
            generator_capacity_excess = np.sum(gens_to_check_s[overloaded_generators] -
                                               gens_limits[overloaded_generators])
            self.capacity_excess = cable_capacity_excess + transformer_capacity_excess + generator_capacity_excess
        else:  # if no components are overloaded
            self.capacity_feasible = True
            self.capacity_excess = 0


class OriginalParameters:
    def __init__(self, de_info):
        # do the power flow calculation
        try:
            pfc_output_data = de_info.pgm.model.calculate_power_flow(max_iterations=50)
            obj_param = ObjectiveParameters(True, de_info, pfc_output_data)
        except power_grid_model.errors.PowerGridError:
            raise ValueError(f"The power flow calculation with the original settings (from the measurement_data.xlsx "
                             f"file) does not converge at {de_info.pgm.measurement_info.date_time}. Please make sure "
                             f"the measurement data is correct so the power flow calculation converges.")

        self.cost = obj_param.value
        self.cost_losses = obj_param.price_losses
        self.feasible = obj_param.feasible
        self.p_pu = pfc_output_data['source']['p'] / de_info.settings.import_limit
        self.q_pu = pfc_output_data['source']['q'] / max(de_info.settings.import_limit, -de_info.settings.export_limit)


class OptimisedParameters:
    def __init__(self, de_info):
        # do the power flow calculation
        de_info.pgm.update_rpc_generators(de_info.x)
        pfc_output_data = de_info.pgm.model.calculate_power_flow(max_iterations=50)
        obj_param = ObjectiveParameters(True, de_info, pfc_output_data)

        self.cost = obj_param.value
        self.cost_losses = obj_param.price_losses
        self.cost_dso_devices = obj_param.price_dso_gens
        self.cost_customer_compensation = obj_param.price_customer_gens
        self.feasible = obj_param.feasible
        self.p_pu = pfc_output_data['source']['p'] / de_info.settings.import_limit
        self.q_pu = pfc_output_data['source']['q'] / max(de_info.settings.import_limit, -de_info.settings.export_limit)
