#! /usr/bin/env python

"""
Flood Simulator (Tian Gan, 2023 Aug)

Description:
This code is created for the participatory modeling project.
It includes a class to run the overland flow process using Landlab components.

Code References:
- https://github.com/gantian127/overlandflow_usecase/blob/master/overland_flow.ipynb
- https://github.com/gregtucker/earthscape_simulator/blob/main/src/evolve_island_world.py
- https://github.com/landlab/landlab/blob/a5b2f68825a36529acd3c451380ec75e9f48e6e3/landlab/grid/voronoi.py#L174

Usage:
method1
from flood_simulator import FloodSimulator
fs = FloodSimulator.from_file('config_file.toml')
fs.run()

method2
$ python flood_simulator.py config_file.toml

"""

import sys
import os

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import rasterio
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from landlab.io import read_esri_ascii, write_esri_ascii
from landlab import imshow_grid
from landlab.components import OverlandFlow, SoilInfiltrationGreenAmpt


class FloodSimulator:
    def __init__(self,
                 terrain,
                 output,
                 model_run,
                 infil_info,
                 olf_info):

        """ Initialize FloodSimulator """

        self.terrain = terrain
        self.output = output
        self.model_run = model_run
        self.infil_info = infil_info
        self.olf_info = olf_info

        self.model_grid = None
        self.outlet_id = None

        self.rain_intensity = None
        self.hydraulic_conductivity = None

        self.setup_grid()

    @classmethod
    def from_file(cls, config_file):
        with open(config_file, mode='rb') as fp:
            args = tomllib.load(fp)
        return cls(**args)

    def setup_grid(self):
        """create RasterModelGrid and add data fields"""

        # DEM field and set boundary condition
        self.model_grid, dem_data = read_esri_ascii(self.terrain['grid_file'],
                                                    name='topographic__elevation')

        if self.terrain['outlet_id'] < 0:
            id_array = self.model_grid.set_watershed_boundary_condition(
                                    dem_data,
                                    nodata_value=self.terrain['nodata_value'],
                                    return_outlet_id=True)
            self.outlet_id = id_array[0]
        else:
            self.outlet_id = self.terrain['outlet_id']
            self.model_grid.set_watershed_boundary_condition_outlet_id(
                                    outlet_id=self.outlet_id,
                                    node_data=dem_data,
                                    nodata_value=self.terrain['nodata_value'])

        # surface water depth TODO: allow tif input
        self.model_grid.add_full("surface_water__depth",
                                 self.olf_info['surface_water_depth'])

        # soil water infiltration depth  TODO: allow tif input
        self.model_grid.add_full("soil_water_infiltration__depth",
                                 self.infil_info['soil_water_infiltration_depth'])

        # maximum surface water depth (this field is added for result analysis)
        self.model_grid.add_zeros('max_surface_water__depth', at='node')

        # maximum discharge (this field is added for result analysis)
        self.model_grid.add_zeros('test_max_discharge', at='node')

        # add rain intensity TODO: allow 3D rain input
        if self.olf_info['rain_file'] != '':
            file = rasterio.open(self.olf_info['rain_file'])
            data = file.read(1).flatten()
            self.rain_intensity = data
        else:
            self.rain_intensity = self.olf_info['rain_intensity']

        # add hydraulic conductivity
        if self.infil_info['conductivity_file'] != '':
            file = rasterio.open(self.infil_info['conductivity_file'])
            flip_data = np.flipud(file.read(1))  # make sure to flip the tiff file data 
            data = flip_data.flatten()
            self.hydraulic_conductivity = data
        else:
            self.hydraulic_conductivity = self.infil_info['hydraulic_conductivity']

    def run(self):
        """
        run overland flow simulation
        """

        # set model run parameters
        model_run_time = self.model_run['model_run_time'] * 60  # duration of run (s)
        storm_duration = self.model_run['storm_duration'] * 60  # duration of rain (s)
        time_step = self.model_run['time_step'] * 60
        elapsed_time = 0.0

        # output setup
        outlet_times = []
        outlet_discharge = []
        output_folder = self.output['output_folder']

        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        # instantiate overland flow component
        overland_flow = OverlandFlow(self.model_grid,
                                     steep_slopes=self.olf_info['steep_slopes'],
                                     alpha=self.olf_info['alpha'],
                                     mannings_n=self.olf_info['mannings_n'],
                                     g=self.olf_info['g'],
                                     theta=self.olf_info['theta'],
                                     )

        # instantiate infiltration component
        if self.model_run['activate_inf']:
            # update parameter values
            for var_name in ['soil_pore_size_distribution_index',
                             'soil_bubbling_pressure',
                             'wetting_front_capillary_pressure_head']:
                if self.infil_info[var_name] is False:
                    self.infil_info[var_name] = None

            # create instance
            infiltration = SoilInfiltrationGreenAmpt(
                self.model_grid,
                hydraulic_conductivity=self.hydraulic_conductivity,
                soil_bulk_density = self.infil_info['soil_bulk_density'],
                initial_soil_moisture_content=self.infil_info[
                    'initial_soil_moisture_content'],
                soil_type=self.infil_info['soil_type'],
                volume_fraction_coarse_fragments= self.infil_info[
                    'volume_fraction_coarse_fragments'],
                coarse_sed_flag=self.infil_info['coarse_sed_flag'],
                surface_water_minimum_depth=self.infil_info[
                    'surface_water_minimum_depth'],
                soil_pore_size_distribution_index=self.infil_info[
                    'soil_pore_size_distribution_index'],
                soil_bubbling_pressure=self.infil_info['soil_bubbling_pressure'],
                wetting_front_capillary_pressure_head=self.infil_info[
                    'wetting_front_capillary_pressure_head']
            )

        # run model simulation
        for time_slice in trange(time_step, model_run_time + time_step, time_step):

            while elapsed_time < time_slice:
                # get adaptive time step
                overland_flow.dt = min(overland_flow.calc_time_step(), time_step)

                # set rainfall intensity
                if elapsed_time < storm_duration:
                    overland_flow.rainfall_intensity = self.rain_intensity/(1000 * 3600)
                else:
                    overland_flow.rainfall_intensity = 0.0

                # run model
                overland_flow.overland_flow(dt=overland_flow.dt)

                if self.model_run['activate_inf']:
                    infiltration.run_one_step(overland_flow.dt)

                # update elapsed time
                elapsed_time += overland_flow.dt

                # get discharge result at outlet
                discharge = overland_flow.discharge_mapper(
                    self.model_grid.at_link["surface_water__discharge"],
                    convert_to_volume=True
                )

                outlet_discharge.append(discharge[self.outlet_id])
                outlet_times.append(elapsed_time)

                # # save the max discharge at each time step (result analysis)
                # self.model_grid.at_node['test_max_discharge'] = np.maximum(
                #     self.model_grid.at_node['test_max_discharge'],
                #     discharge
                # )

            # # save surface water depth at each time step
            # write_esri_ascii(os.path.join(output_folder,
            #                  "water_depth_{}.asc".format(time_slice)),
            #                  self.model_grid, 'surface_water__depth', clobber=True)

            # # save the max water depth at each time step
            # self.model_grid.at_node['max_surface_water__depth'] = np.maximum(
            #     self.model_grid.at_node['max_surface_water__depth'],
            #     self.model_grid.at_node['surface_water__depth'])

            # plot overland flow results
            if self.output['plot_olf']:
                fig, ax = plt.subplots(
                    2, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [1, 1.5]}
                )
                fig.suptitle("Results at {} min".format(time_slice / 60))

                ax[0].plot(outlet_times, outlet_discharge, "-")
                ax[0].set_xlabel("Time elapsed (s)")
                ax[0].set_ylabel("discharge (cms)")
                ax[0].set_title("Water discharge at the outlet")

                imshow_grid(
                    self.model_grid,
                    "surface_water__depth",
                    cmap="Blues",
                    var_name="surface water depth (m)",
                )
                ax[1].set_title("")
                ax[1].set_xlabel('east-west distance (m)')
                ax[1].set_ylabel('north-south distance (m)')

                plt.close(fig)
                fig.savefig(os.path.join(output_folder, f"flow_{time_slice}.png"))

            # plot infiltration result
            if self.model_run['activate_inf'] and self.output['plot_inf']:
                fig, ax = plt.subplots(figsize=(8, 6))
                imshow_grid(
                    self.model_grid,
                    "soil_water_infiltration__depth",
                    cmap="Blues",
                    var_name=" soil water infiltration depth (m)",
                )
                ax.set_title("")
                ax.set_xlabel('east-west distance (m)')
                ax.set_ylabel('north-south distance (m)')

                plt.close(fig)
                fig.savefig(os.path.join(output_folder, f"infil_{time_slice}.png"))

        # save outlet discharge
        outlet_result = pd.DataFrame(list(zip(outlet_times, outlet_discharge)),
                                     columns=['time', 'discharge'])
        outlet_result.to_csv(os.path.join(
            self.output['output_folder']
            if os.path.isdir(self.output['output_folder']) else os.getcwd(),
            'outlet_discharge.csv')
        )

        # # save max surface water depth
        # max_depth = self.model_grid.at_node['max_surface_water__depth']
        # max_depth[max_depth == 1e-12] = 0

        # # as csv
        # df = pd.DataFrame(max_depth, columns=['z_value'])
        # df.to_csv(os.path.join(
        #     self.output['output_folder']
        #     if os.path.isdir(self.output['output_folder']) else os.getcwd(),
        #     'max_water_depth.csv')
        # )

        # # as ascii
        # write_esri_ascii(os.path.join(output_folder, "max_water_depth.asc"),
        #                  self.model_grid, 'max_surface_water__depth', clobber=True)

        # write_esri_ascii(os.path.join(output_folder, "max_discharge.asc"),
        #                  self.model_grid, 'test_max_discharge', clobber=True
        #                  )



if __name__ == "__main__":
    """
    Launch a model run for flood simulator. 
    Command-line argument is the path of a configuration file (toml-format). 
    """

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        fs = FloodSimulator.from_file(config_file)
        fs.run()
    else:
        print('Please provide a configuration file path to run the model.')

