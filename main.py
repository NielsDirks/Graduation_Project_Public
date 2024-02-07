from optimisation_settings import OptimisationSettings
from measurement_data import find_measurement_row_idxs
from DE import differential_evolution
from plotting import OutputData

optimisation_settings = OptimisationSettings()

measurement_range = find_measurement_row_idxs(optimisation_settings)

if not measurement_range:   # if there is no measurement_data.xlsx file generate one via:
    result = differential_evolution(settings=optimisation_settings,
                                    measurement_idx=measurement_range)
else:   # run the optimisation
    # initialise object for storing and plotting PQ-points in the end
    output_data = OutputData(optimisation_settings)
    for measurement_idx in range(measurement_range[0], measurement_range[1]+1):
        # optimise for one time instance given by measurement_idx
        result = differential_evolution(settings=optimisation_settings,
                                        measurement_idx=measurement_idx)

        # store output data in the PQ-point arrays
        output_data.store_data(result)

        # print the optimisation result for this time step
        print(f"The solution of the optimisation for {result.date_time} is {result.fun}.")
        print(f"{result.nfev} load flow calculations were performed.")
        print("List of generator names, Vision node IDs and names where the generator is connected respectively:")
        print("Note: a positive value for reactive power indicates inductive reactive power and is thus consumed by "
              "the generator.\nA negative value indicates capacitive reactive power and is produced by the generator.")
        # get widths of to be printed text for printing a readable output
        max_width_gen_name = max(len(str(name)) for name in result.vision_info.rpc_generator_name)
        max_width_node_id = max(len(str(node_id)) for node_id in result.vision_info.rpc_node_vision_ids)
        max_width_node_name = max(len(str(name)) for name in result.vision_info.rpc_node_vision_names)
        for gen in range(len(result.x)):
            print(f"Generator name: {result.vision_info.rpc_generator_name[gen].ljust(max_width_gen_name+3)}"
                  f"Vision node ID: {result.vision_info.rpc_node_vision_ids[gen].ljust(max_width_node_id+3)}"
                  f"Vision node name: {result.vision_info.rpc_node_vision_names[gen].ljust(max_width_node_name+3)}"
                  f"Reactive power: {int(round(result.x[gen],-3)/1000)} kvar")
        print("\n\n")

    # when all time steps have been optimised, save output data and plot result
    print(f"A total of {output_data.total_function_evaluations} load flow calculations were performed.")
    output_data.save_data(optimisation_settings)
    output_data.plot_data()
