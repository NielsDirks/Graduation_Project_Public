import pandas as pd
from datetime import datetime


def find_measurement_row_idxs(settings):
    file_path = f'MV_Grids/{settings.mv_grid}/measurement_data.xlsx'
    # Read the Excel file
    try:
        df = pd.read_excel(file_path)
        df = df.iloc[:, 0]

        # find the first row with data
        first_row_idx = 0
        for i in range(len(df)):
            if isinstance(df.iloc[i], datetime):
                first_row_idx = i
                break

        # Convert the date column to datetime format
        df[first_row_idx:] = pd.to_datetime(df[first_row_idx:])

        # Parse start and end dates
        start_date = pd.to_datetime(settings.start_date, format='%d-%m-%Y')
        end_date = pd.to_datetime(settings.end_date + ' 23:59:59', format='%d-%m-%Y %H:%M:%S')

        # Check for correctness of start and end date
        if end_date < start_date:
            raise Exception("Please input an end date that is later in time than the start date!")
        df_dates = pd.to_datetime(df[first_row_idx:]).dt.date
        if start_date.date() not in df_dates.values:
            raise Exception("The start date is not found in the measurement data. Please select a correct start date, "
                            "or get the correct measurement data into the measurement_data.xlsx file!")
        if end_date.date() not in df_dates.values:
            raise Exception("The end date is not found in the measurement data. Please select a correct end date, or "
                            "get the correct measurement data into the measurement_data.xlsx file!")

        # Filter rows based on the given start and end date
        filtered_rows = (df.iloc[first_row_idx:] >= start_date) & (df.iloc[first_row_idx:] <= end_date)
        filtered_rows = filtered_rows.index[filtered_rows].tolist()

        # Get start index and end index
        start_idx = int(min(filtered_rows))
        end_idx = int(max(filtered_rows))

        # Return start and end row index
        return start_idx, end_idx
    except FileNotFoundError:
        return False


def generate_measurement_file(pgm):
    # get and create paths for various files
    mv_grid = pgm.optimisation_settings.mv_grid
    grid_file_path = f'MV_Grids/{mv_grid}/grid.xlsx'
    measurement_data_path = f'MV_Grids/{mv_grid}/measurement_data.xlsx'

    # get info from power grid model
    try:
        load_pgm_node_ids = set(pgm.load_nodes)
    except TypeError:
        load_pgm_node_ids = set([])
    try:
        gen_pgm_node_ids = set(pgm.gen_nodes_orig)
    except TypeError:
        gen_pgm_node_ids = set([])
    node_info = pgm.extra_info

    # do some set theory to get necessary info
    unique_pgm_node_ids = list(load_pgm_node_ids.union(gen_pgm_node_ids))
    gen_only_pgm_node_ids = list(gen_pgm_node_ids - load_pgm_node_ids)

    # Get Vision node numbers ('Nummer' in Vision 'Knooppunten' table)
    unique_vision_knooppunt_nummers = []
    node_info = [node_info[node] for node in unique_pgm_node_ids]
    for entry in node_info:
        unique_vision_knooppunt_nummers.append(entry['id_reference']['key']['Nummer'])

    # Get 'Korte naam' info from the Vision Excel export
    df = pd.read_excel(grid_file_path, sheet_name='Knooppunten', na_values=['']).fillna('')
    nummer_column = df['Nummer'].values
    korte_naam_column = df['Korte naam'].values
    mapping_dict = dict(zip(nummer_column, korte_naam_column))
    unique_nodes_korte_naam = [mapping_dict[num] for num in unique_vision_knooppunt_nummers]

    # Check for empty korte naam cells, if one or more are present, raise an error
    empty_cell_check = [entry == '' for entry in unique_nodes_korte_naam]
    no_korte_naam_knooppunt_nummers = [value for value, is_empty in
                                       zip(unique_vision_knooppunt_nummers, empty_cell_check) if is_empty]
    if no_korte_naam_knooppunt_nummers:     # raise error if no_korte_naam_knooppunt_nummers contains value(s)
        raise Exception(f"Not all medium voltage substations have a 'Korte naam' filled in. Please enter a 'Korte "
                        f"naam' for the Vision nodes with 'Vision Knooppunt Nummer' {no_korte_naam_knooppunt_nummers}")

    # Now make the header for the measurement_data.xlsx file
    # duplicate every entry
    header = [entry for entry in unique_nodes_korte_naam for _ in range(2)]
    # add ' P' to the first copy and ' Q' to the second
    header = [f"{entry} P" if i % 2 == 0 else f"{entry} Q" for i, entry in enumerate(header)]
    header.insert(0, 'DATUMTIJD')

    # create measurement_data sheet, this sheet should contain all unique nodes with P and Q behind it
    measurement_data_sheet = pd.DataFrame(columns=header)

    # create appliances and pgm_appliance_ids for mapping measurements to appliances
    appliances = ['sym_gen' if pgm_node_id in gen_only_pgm_node_ids
                  else 'sym_load' for pgm_node_id in unique_pgm_node_ids]

    # Initialize an empty list to store appliance ids
    pgm_appliance_ids = []

    # Iterate over each appliance in the list with appliances ('sym_gen' or 'sym_load')
    for appliance, pgm_appliance_id in zip(appliances, unique_pgm_node_ids):
        appliance_array = pgm.input_data[appliance]
        match_array = pgm.input_data[appliance]['node'] == pgm_appliance_id
        appliance_data = appliance_array[match_array]
        pgm_appliance_ids.append(appliance_data['id'][0])

    # create 'mapping' sheet were Vision Korte naam is mapped to the pgm node id and to which appliance measurement
    # data should be assigned to
    index = ['Vision Korte naam', 'PGM node id', 'Appliance', 'PGM appliance id']
    mapping_sheet = pd.DataFrame(data=[unique_nodes_korte_naam, unique_pgm_node_ids, appliances, pgm_appliance_ids],
                                 index=index)

    # Now create the file with the two sheets
    writer = pd.ExcelWriter(measurement_data_path)
    measurement_data_sheet.to_excel(writer, sheet_name='data', index=False)
    mapping_sheet.to_excel(writer, sheet_name='mapping', index=True, header=False)
    writer.close()

    raise Exception(f"No measurement_data.xlsx file was present in the folder named {mv_grid} in the MV_Grids folder. "
                    f"However, this file is now created. Please fill this file with corresponding measurement data (in "
                    f"[kW] and [kvar]) and run the program again!")


class MeasurementInfo:
    def __init__(self, pgm):
        # get optimisation settings
        mv_grid = pgm.optimisation_settings.mv_grid
        self.measurement_row_idx = pgm.measurement_idx
        self.measurement_data_path = f'MV_Grids/{mv_grid}/measurement_data.xlsx'

        # get and set active and reactive power measurement data
        self._get_measurement_data()

        # get and set mapping info
        self._get_mapping_info()

        # Remove unnecessary entries
        del self.measurement_row_idx, self.measurement_data_path

    def _get_measurement_data(self):
        df = pd.read_excel(self.measurement_data_path, sheet_name='data')
        measurement_data = df.iloc[self.measurement_row_idx]

        # for validation in Vision convenience
        self.date_time = pd.to_datetime(measurement_data['DATUMTIJD']).strftime('%d-%m-%Y %H:%M:%S')
        print(f"\nMeasurement input data for {self.date_time}:")
        print(f"{measurement_data.to_string(index=True)}")
        print("Note: measurement input data is always assigned to a load at the node. Only when no load is present, "
              "the input data is assigned to a generator.\nFor validation in Vision, signs should be flipped when "
              "assigning input data to a generator.\n")

        # separate in active and reactive power data
        self.active_power_data = measurement_data[measurement_data.index.str.endswith('P')].values * 1000      # [W]
        self.reactive_power_data = measurement_data[measurement_data.index.str.endswith('Q')].values * 1000    # [var]

    def _get_mapping_info(self):
        df = pd.read_excel(self.measurement_data_path, sheet_name='mapping', index_col=0)
        self.appliance = df.loc['Appliance'].values
        self.pgm_appliance_id = df.loc['PGM appliance id'].values
