import pandas as pd
import numpy as np

from power_grid_model_io.converters import VisionExcelConverter
from power_grid_model import PowerGridModel, CalculationType, initialize_array, LoadGenType
from power_grid_model.validation import assert_valid_input_data, assert_valid_batch_data

from measurement_data import generate_measurement_file, MeasurementInfo


class PGM:

    def __init__(self, de_info):
        self.optimisation_settings = de_info.settings
        self.measurement_idx = de_info.measurement_idx

        grid_file_path = f'MV_Grids/{self.optimisation_settings.mv_grid}/grid.xlsx'

        # Read Vision file
        converter = VisionExcelConverter(source_file=grid_file_path, language="nl")
        self.input_data, self.extra_info = converter.load_input_data()
        # validate and initialise PGM grid
        assert_valid_input_data(self.input_data, calculation_type=CalculationType.power_flow, symmetric=True)
        self.model = PowerGridModel(input_data=self.input_data)

        # find the node id where the source (TenneT) is connected. This node is excluded for reactive power control
        self.source_node = self.input_data['source']['node']

        # initialise the total number of generators and the number of DSO generators
        self.n_gens = 0
        self.n_dso_gens = 0

        try:
            # find the total number of (original) generators and obtain node ids where these are connected
            self.n_gen_orig = self.model.all_component_count['sym_gen']
            self.gen_nodes_orig = np.unique(self.input_data['sym_gen']['node'])
            self.gen_ids_orig = self.input_data['sym_gen']['id']
        except KeyError:
            self.n_gen_orig = None
            self.gen_nodes_orig = None
            self.gen_ids_orig = None

        try:
            # find the total number of loads and obtain node ids where these are connected
            self.n_load = self.model.all_component_count['sym_load']
            self.load_nodes = np.unique(self.input_data['sym_load']['node'])
            self.load_ids = self.input_data['sym_load']['id']
        except KeyError:
            self.n_load = None
            self.load_nodes = None
            self.load_ids = None

        # If strategy is dso_only or both, generators at every reactive power controllable node should be added
        # Note that generators are chosen, not loads. In this case capacitive reactive power is positive (delivered by
        # generator), inductive reactive power is negative (consumed by generator). 'input_data', 'extra_info' and
        # 'model' will be updated in this function
        self.rpc_nodes = None
        self.rpc_gen_ids = None
        if self.optimisation_settings.appliance_strategy in ('dso_only', 'both'):
            self._create_dso_generators()

        # update p and q input of generators and loads with the given measurement data for gens and loads
        # dso_loads are not affected
        # Note that only self.model is updated by pgm, but not self.input_data
        self._initial_update()

        # initialise limits and integrality
        self.limits = None
        self.integrality = None

        # initialise vision_info
        self.vision_info = self.VisionInfo()

        # set original generator capacity [kVA] limits
        self._get_generator_limits(grid_file_path)

    def _create_dso_generators(self):
        # determine the starting id of new components (every component in pgm has a unique id)
        new_id = len(self.extra_info)

        # determine which nodes have 4 or more cables connected to it
        from_nodes = self.input_data['line']['from_node']
        to_nodes = self.input_data['line']['to_node']
        nodes = np.concatenate((from_nodes, to_nodes))
        unique_nodes, counts = np.unique(nodes, return_counts=True)
        self.rpc_nodes = unique_nodes[counts >= 4]

        # determine how many generators should be added (number of nodes - 1 for source node)
        self.n_dso_gens = len(self.rpc_nodes)

        # for every node where 4 or more cables are connected, create a generator where reactive power can be controlled
        if self.n_dso_gens:
            dso_gens = initialize_array('input', 'sym_gen', self.n_dso_gens)
            dso_gens['id'] = np.arange(new_id, new_id + self.n_dso_gens)
            dso_gens['node'] = self.rpc_nodes
            dso_gens['status'] = [1] * self.n_dso_gens
            dso_gens['type'] = [LoadGenType.const_power] * self.n_dso_gens
            dso_gens['p_specified'] = [0] * self.n_dso_gens
            dso_gens['q_specified'] = [0] * self.n_dso_gens
        else:
            raise ValueError("No nodes were found where reactive power flow can be controlled. Make sure there is at "
                             "least 1 node were 4 or more cables or links are connected.")

        # store the ids of the generators where reactive power can be controlled
        self.rpc_gen_ids = dso_gens['id']

        # update 'input_data'
        try:
            self.input_data['sym_gen'] = np.concatenate((self.input_data['sym_gen'], dso_gens),
                                                        dtype=self.input_data['sym_gen'].dtype)
        except KeyError:
            self.input_data['sym_gen'] = dso_gens

        # update 'extra_info'
        for i in range(self.n_dso_gens):
            self.extra_info[new_id+i] = {'id_reference': {
                'table': 'DSO generator, added by algorithm',
                'key': {'Knooppunt.Nummer': self.rpc_nodes[i]}},    # , 'Subnummer': 1
                'Naam': f'DSO Generator {i+1}'}

        # validate and initialise PGM grid
        assert_valid_input_data(self.input_data, calculation_type=CalculationType.power_flow, symmetric=True)
        self.model = PowerGridModel(input_data=self.input_data)

    def _initial_update(self):
        """
        This function sets a value to generators and loads connected to MV stations in the grid. This value does not
        change during the optimisation. (Only if the generator is used for controlling reactive power.)
        If the measurement_data.xlsx file does not exist (then self.measurement_idx will be false) a new file (named
        measurement_data.xlsx) will be created with a column for the P and Q measurements for each MV substation that
        needs to be filled with data from measurements.
        """

        # if measurement_data.xlsx does not exist, make the file and terminate program
        if isinstance(self.measurement_idx, bool):
            generate_measurement_file(self)

        # get measurement info from measurement file
        self.measurement_info = MeasurementInfo(self)

        # divide info into generator and load arrays
        gen_info_idx = np.where(self.measurement_info.appliance == 'sym_gen')[0]
        update_gen_ids = self.measurement_info.pgm_appliance_id[gen_info_idx]
        update_gen_p = -self.measurement_info.active_power_data[gen_info_idx]    # change sign for generator
        update_gen_q = -self.measurement_info.reactive_power_data[gen_info_idx]  # change sign for generator
        load_info_idx = np.where(self.measurement_info.appliance == 'sym_load')[0]
        update_load_ids = self.measurement_info.pgm_appliance_id[load_info_idx]
        update_load_p = self.measurement_info.active_power_data[load_info_idx]
        update_load_q = self.measurement_info.reactive_power_data[load_info_idx]

        # initialise update_data array
        update_data = {}

        # first set active and reactive power of all generators to 0 if there are generators
        if self.n_gen_orig:
            update_sym_gen = initialize_array("update", "sym_gen", self.n_gen_orig)
            update_sym_gen['id'] = self.gen_ids_orig
            update_sym_gen['p_specified'] = [0] * self.n_gen_orig
            update_sym_gen['q_specified'] = [0] * self.n_gen_orig

            # then update all generators that will be assigned measurement data
            for i, gen_id in enumerate(update_gen_ids):
                idx = np.where(update_sym_gen['id'] == gen_id)[0][0]
                update_sym_gen[idx]['p_specified'] = update_gen_p[i]
                update_sym_gen[idx]['q_specified'] = update_gen_q[i]

            update_data['sym_gen'] = update_sym_gen
            # Also save the original setpoints for Q for all customer generators. This is used in objective_parameters.py
            self.settings_gens_orig = update_sym_gen

        # update all loads
        if self.n_load:
            update_sym_load = initialize_array("update", "sym_load", self.n_load)
            update_sym_load['id'] = update_load_ids
            update_sym_load['p_specified'] = update_load_p
            update_sym_load['q_specified'] = update_load_q

            update_data['sym_load'] = update_sym_load

        # validate updated input data
        assert_valid_batch_data(input_data=self.input_data, update_data=update_data,
                                calculation_type=CalculationType.power_flow, symmetric=True)
        # update model, updates are not visible in input_data
        self.model.update(update_data=update_data)

    def _get_generator_limits(self, grid_file):
        try:
            df = pd.read_excel(grid_file, sheet_name='Synchrone generatoren')
            self.generators_capacity_limit = np.array(df['Snom'].iloc[1:].to_numpy()) * 1000    # [VA]
            self.generators_min_q = np.array(df['Qmin'].iloc[1:].to_numpy()) * 1000              # [var]
            self.generators_max_q = np.array(df['Qmax'].iloc[1:].to_numpy()) * 1000              # [var]
            # if min_q and/or max_q contains 0, replace it with the maximum capacity of the generator
            self.generators_min_q[self.generators_min_q == 0] = \
                -self.generators_capacity_limit[self.generators_min_q == 0]
            self.generators_max_q[self.generators_max_q == 0] = \
                self.generators_capacity_limit[self.generators_max_q == 0]
        except ValueError:
            self.generators_capacity_limit = []
            self.generators_min_q = []
            self.generators_max_q = []

    def _set_dso_limits(self):
        dso_gen_limit = self.optimisation_settings.dso_gen_lim
        dso_gen_step_size = self.optimisation_settings.dso_gen_step_size

        # set limits of the dso generators to the minimum and maximum step of the dso generator
        # note: negative step is capacitor, positive step is inductor
        min_val = 0     # -int(dso_gen_limit/dso_gen_step_size)
        max_val = int(dso_gen_limit / dso_gen_step_size)
        # number of rpc nodes in self.rpc_nodes is equal to the number of dso generators in the grid
        self.limits = [(min_val, max_val) for _ in self.rpc_nodes]
        # set the integrality for dso generators to true (steps of a cap/ind bank must be an integer)
        try:
            self.integrality.extend([True] * len(self.limits))
        except AttributeError:  # if self.integrality is still set to None, create array:
            self.integrality = [True] * len(self.limits)

    def _set_customers_limits(self):
        try:
            min_vals = self.generators_min_q
            max_vals = self.generators_max_q
            if not self.limits:     # if self.limits was not set yet (strategy == 'customers_only')
                self.limits = [(min_vals[gen], max_vals[gen]) for gen in range(self.n_gen_orig)]
                # the ids of the generators that are able to control reactive power are stored in self.rpc_gen_ids
                self.rpc_gen_ids = self.gen_ids_orig
                # set self.rpc_nodes to contain every node where reactive power will be controlled
                self.rpc_nodes = self.input_data['sym_gen']['node']
                # leave self.integrality set to None
            else:   # if self.limits already exists, extend it (strategy == 'both')
                self.limits = self.limits + [(min_vals[gen], max_vals[gen]) for gen in range(self.n_gen_orig)]
                # Create a temporary array containing only the generators of customers. As the dso generators have
                # already been created with the _create_dso_generators() function, remove these to create the temporary
                # array. self.rpc_gen_ids only contains the dso generator ids yet, so this can be used to select which
                # entries should be removed
                customer_only_gens = self.input_data['sym_gen'][~np.isin(self.input_data['sym_gen']['id'],
                                                                         self.rpc_gen_ids)]
                # add the ids of all customers generators to the existing rpc_gen_ids array (that contains ids for all
                # dso generators that were created using the _create_dso_generators() function)
                self.rpc_gen_ids = np.append(self.rpc_gen_ids, customer_only_gens['id'])
                # update self.rpc_nodes to contain every node where reactive power will be controlled
                self.rpc_nodes = np.append(self.rpc_nodes, customer_only_gens['node'])
                # add entries for customer generators to the integrality array
                self.integrality.extend([False] * len(customer_only_gens))
        except TypeError:
            raise TypeError("No customers with generators were found. Please add generators to the grid, or use a "
                            "different appliance strategy.")

    def get_bounds(self):
        if self.optimisation_settings.appliance_strategy == 'dso_only':
            # set dso generator limits and integrality
            self._set_dso_limits()
        elif self.optimisation_settings.appliance_strategy == 'customers_only':
            # set customer generator limits
            self._set_customers_limits()
        elif self.optimisation_settings.appliance_strategy == 'both':
            # set dso generator limits and integrality
            self._set_dso_limits()
            # set customer generator limits and integrality
            self._set_customers_limits()
        else:
            raise ValueError("Please select a valid appliance strategy")

        # determine the total number of generators
        self.n_gens = len(self.rpc_gen_ids)

        # update self.pgm.vision_info to present the user with useful data after the optimisation has finished
        self.vision_info._set_vision_node_info(self)

        return self.limits, self.integrality

    def update_rpc_generators(self, parameters):
        """
        Function to update the reactive power output of the rpc generators
        Note that a positive parameter indicates reactive power consumption (inductive) and a negative parameter
        indicates reactive power production (capacitive). However, signs should be swapped for generators, where a
        positive sign indicates production and a negative sign indicates consumption. Multiply the parameter of DSO
        generators with its given step size (DSO generators are initialised before customer generators, therefore,
        update parameter 0 to n_dso_gens first).
        """
        parameters = -parameters
        if self.n_dso_gens:
            parameters[:self.n_dso_gens] = parameters[:self.n_dso_gens] * self.optimisation_settings.dso_gen_step_size
        update_sym_gen = initialize_array('update', 'sym_gen', self.n_gens)
        update_sym_gen['id'] = self.rpc_gen_ids
        # leave active power at zero, so no need to specify
        update_sym_gen['q_specified'] = parameters

        update_data = {'sym_gen': update_sym_gen}
        self.model.update(update_data=update_data)

        # # validate the data
        # validate_batch_data(input_data=self.input_data, update_data=update_data,
        #                     calculation_type=CalculationType.power_flow, symmetric=True)

    class VisionInfo:
        def __init__(self):
            self.rpc_generator_name = []
            self.rpc_node_vision_ids = []
            self.rpc_node_vision_names = []

        def _set_vision_node_info(self, pgm):
            # get the names of the dso and/or customer generators from extra info
            try:
                rpc_gen_info = [pgm.extra_info[gen] for gen in pgm.rpc_gen_ids]
                for entry in rpc_gen_info:
                    try:
                        gen_name = entry['Naam']
                    except KeyError:
                        gen_name = 'Name unknown'
                    self.rpc_generator_name.append(gen_name)
            except TypeError:
                pass

            # get the rpc node vision names and ids from extra info
            try:
                nodes_info = [pgm.extra_info[node] for node in pgm.rpc_nodes]
                for entry in nodes_info:
                    try:
                        vision_id = entry['ID']
                    except KeyError:
                        vision_id = 'ID unknown'
                    self.rpc_node_vision_ids.append(vision_id)

                    try:
                        vision_name = entry['Naam']
                    except KeyError:
                        vision_name = 'Name unknown'
                    self.rpc_node_vision_names.append(vision_name)
            except TypeError:
                pass
