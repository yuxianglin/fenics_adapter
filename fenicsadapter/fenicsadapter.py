"""This module handles CustomExpression and initalization of the FEniCS
adapter.

:raise ImportError: if PRECICE_ROOT is not defined
"""
import dolfin
from dolfin import UserExpression, SubDomain, File, info, mpi_comm_world, MPI, vertex_to_dof_map, facets, vertices
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
import numpy as np
from .config import Config
try:
    import precice
except ImportError:
    import os
    import sys
    # check if PRECICE_ROOT is defined
    if not os.getenv('PRECICE_ROOT'):
       raise Exception("ERROR: PRECICE_ROOT not defined!")

    precice_root = os.getenv('PRECICE_ROOT')
    precice_python_adapter_root = precice_root+"/src/precice/bindings/python"
    sys.path.insert(0, precice_python_adapter_root)
    import precice



class Adapter:
    """Initializes the Adapter. Initalizer creates object of class Config (from
    config.py module).

    :ivar _config: object of class Config, which stores data from the JSON config file
    """
    def __init__(self, adapter_config_filename='precice-adapter-config.json'):

        self._config = Config(adapter_config_filename)

        self._solver_name = self._config.get_solver_name()

        self.comm=mpi_comm_world()
        self.rank=MPI.rank(self.comm)
        self.size=MPI.size(self.comm)

        self._interface = precice.Interface(self._solver_name, self.rank, self.size)
        self._interface.configure(self._config.get_config_file_name())
        self._dimensions = self._interface.get_dimensions()

        self._coupling_subdomain = None # initialized later
        self._V_fenics = None # initialized later

        ## coupling mesh related quantities
        self._coupling_mesh_vertices = None # initialized later
        self._mesh_name = self._config.get_coupling_mesh_name()
        self._mesh_id = self._interface.get_mesh_id(self._mesh_name)
        self._vertex_ids = None # initialized later

        ## write data related quantities (write data is written by this solver to preCICE)
        self._write_data_name = self._config.get_write_data_name()
        self._write_data_id = self._interface.get_data_id(self._write_data_name, self._mesh_id)
        self._write_data = None

        ## read data related quantities (read data is read by this solver from preCICE)
        self._read_data_name = self._config.get_read_data_name()
        self._read_data_id = self._interface.get_data_id(self._read_data_name, self._mesh_id)
        self._read_data = None

        ## numerics
        self._precice_tau = None

        ## checkpointing

    def convert_fenics_to_precice(self, data, dofs):
        """Converts FEniCS data of type dolfin.Function into
        Numpy array for all x and y coordinates on the boundary.

        :param data: FEniCS boundary function
        :raise Exception: if type of data cannot be handled
        :return: array of FEniCS function values at each point on the boundary
        """
        if type(data) is dolfin.Function:
            return data.vector()[dofs]
        elif type(data) is dolfin.cpp.la.Vector:
            return data.get_local()[dofs]
    def extract_coupling_boundary_vertices1(self, function_space):
        """Extracts verticies which lay on the boundary. Currently handles 2D
        case properly, 3D is circumvented.

        :raise Exception: if no correct coupling interface is defined
        :return: stack of verticies
        """
        n = 0
        local_dofs=[]
        vertices_x = []
        vertices_y = []
        if self._dimensions == 3:
            vertices_z = []
        con=[]

        if not issubclass(type(self._coupling_subdomain), SubDomain):
            raise Exception("no correct coupling interface defined!")
       
        mesh=function_space.mesh()
        v2d=vertex_to_dof_map(function_space)
        value_size=function_space.ufl_element().value_size()
        for f in facets(mesh):
            interface=True
            #if f.exterior():
            for v in vertices(f):
                if self._dimensions==2:
                    if not self._coupling_subdomain.inside([v.x(0), v.x(1)], True):
                        interface=False
                elif self._dimensions==3:
                    if not self._coupling_subdomain.inside([v.x(0), v.x(1), v.x(2)], True):
                        interface=False
            #else:
            #    interface=False
            if interface:
                for v in vertices(f):
                    for ii in range(value_size):
                        local_dof=v2d[value_size*v.index()+ii]
                        if ii==0:
                            con.append(local_dof)
                        if local_dof not in local_dofs:
                            local_dofs.append(local_dof)
                            if ii==0:
                                n+=1
                                vertices_x.append(v.x(0))
                                vertices_y.append(v.x(1))
                                if self._dimensions == 3:
                                    vertices_z.append(v.x(2))



        if self._dimensions == 2:
            return np.column_stack([vertices_x, vertices_y]), n, local_dofs, con
        elif self._dimensions == 3:
            return np.column_stack([vertices_x, vertices_y, vertices_z]), n, local_dofs, con

    def extract_coupling_boundary_vertices(self, function_space):
        """Extracts verticies which lay on the boundary. Currently handles 2D
        case properly, 3D is circumvented.

        :raise Exception: if no correct coupling interface is defined
        :return: stack of verticies
        """
        n = 0
        local_dofs=[]
        vertices_x = []
        vertices_y = []
        if self._dimensions == 3:
            vertices_z = []

        if not issubclass(type(self._coupling_subdomain), SubDomain):
            raise Exception("no correct coupling interface defined!")
        
        dofs=function_space.dofmap().dofs()#global_dof
        value_size=function_space.ufl_element().value_size()
        coords=function_space.tabulate_dof_coordinates().reshape(len(dofs), self._dimensions)[range(0, len(dofs), value_size)]
        for ii, coord in enumerate(coords):#ii is local dof
            if self._coupling_subdomain.inside(coord, True):
                n+=1
                for jj in range(value_size):
                    local_dofs.append(value_size*ii + jj)
                vertices_x.append(coord[0])
                vertices_y.append(coord[1])
                if self._dimensions == 3:
                    vertices_z.append(coord[2])


        if self._dimensions == 2:
            return np.column_stack([vertices_x, vertices_y]), n, local_dofs, []
        elif self._dimensions == 3:
            return np.column_stack([vertices_x, vertices_y, vertices_z]), n, local_dofs, []

    def set_coupling_mesh(self, read_function_space, write_function_space, subdomain, mapping):
        """Sets the coupling mesh. Called by initalize() function at the
        beginning of the simulation.
        """
        self._coupling_subdomain = subdomain
        #self._V_fenics = function_space
        self.read_coupling_mesh_vertices,  self.read_n_vertices,  self.read_local_dofs, con= self.extract_coupling_boundary_vertices1(read_function_space)
        self.write_coupling_mesh_vertices, self.write_n_vertices, self.write_local_dofs, con=self.extract_coupling_boundary_vertices1(write_function_space)
        assert(self.read_coupling_mesh_vertices.shape[0]==self.read_n_vertices)
        assert(self.write_coupling_mesh_vertices.shape[0]==self.write_n_vertices)
        assert(self.read_coupling_mesh_vertices.shape[1]==self._dimensions)
        assert(self.write_coupling_mesh_vertices.shape[1]==self._dimensions)
        if self.read_n_vertices!=self.write_n_vertices:
            raise Exception("read data and write data should have at least same order!")
        if mapping:
            self.read_coupling_mesh_vertices=mapping(self.read_coupling_mesh_vertices)
        self.read_vertex_ids = np.zeros(self.read_n_vertices)
        self.write_vertex_ids = np.zeros(self.write_n_vertices)
        if self.read_n_vertices>0:
            self._interface.set_mesh_vertices(self._mesh_id, self.read_n_vertices, self.read_coupling_mesh_vertices.flatten(), self.read_vertex_ids)
            self.write_vertex_ids=self.read_vertex_ids
            if con:
                numElement=len(con)//3
                for ii in range(numElement):
                    v1=con[3*ii]
                    v2=con[3*ii+1]
                    v3=con[3*ii+2]
                    ind1=self.write_local_dofs.index(v1)
                    ind2=self.write_local_dofs.index(v2)
                    ind3=self.write_local_dofs.index(v3)
                    self._interface.set_mesh_triangle_with_edges(self._mesh_id, self.read_vertex_ids[ind1], self.read_vertex_ids[ind2], self.read_vertex_ids[ind3])

    def set_write_field(self, write_function_init):
        """Sets the write field. Called by initalize() function at the
        beginning of the simulation.

        :param write_function_init: function on the write field
        """
        self._write_data = self.convert_fenics_to_precice(write_function_init, self.write_local_dofs)
        #self._write_data = self.convert_fenics_to_precice(write_function_init)

    def set_read_field(self):
        """Sets the read field. Called by initalize() function at the
        beginning of the simulation.

        """
        self._read_data = np.zeros(len(self.read_local_dofs))

    def create_coupling_boundary_condition(self):
        """Creates the coupling boundary conditions using an actual implementation CustomExpression."""
        x_vert, y_vert = self.extract_coupling_boundary_coordinates()

        try:  # works with dolfin 1.6.0
            self._coupling_bc_expression = self._my_expression()
        except (TypeError, KeyError):  # works with dolfin 2017.2.0
            self._coupling_bc_expression = self._my_expression(degree=0)
        self._coupling_bc_expression.set_boundary_data(self._read_data, x_vert, y_vert)

    def create_coupling_dirichlet_boundary_condition(self, function_space):
        """Creates the coupling Dirichlet boundary conditions using
        create_coupling_boundary_condition() method.

        :return: dolfin.DirichletBC()
        """
        self.create_coupling_boundary_condition()
        return dolfin.DirichletBC(function_space, self._coupling_bc_expression, self._coupling_subdomain)

    def create_coupling_neumann_boundary_condition(self, test_functions):
        """Creates the coupling Neumann boundary conditions using
        create_coupling_boundary_condition() method.

        :return: expression in form of integral: g*v*ds. (see e.g. p. 83ff
         Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The
         FEniCS Tutorial Volume I." (2016).)
        """
        self.create_coupling_boundary_condition()
        return self._coupling_bc_expression * test_functions * dolfin.ds  # this term has to be added to weak form to add a Neumann BC (see e.g. p. 83ff Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The FEniCS Tutorial Volume I." (2016).)

    def advance(self, write_function, read_function, dt, T=None,  init_data1=None, init_data2=None, init_data3=None):
    #def advance(self, write_function, read_function, u_n, t, dt, n):
        """Calls preCICE advance function using precice and manages checkpointing.
        The solution u_n is updated by this function via call-by-reference. The corresponding values for t and n are returned.

        This means:
        * either, the checkpoint self._u_cp is assigned to u_n to repeat the iteration,
        * or u_n+1 is assigned to u_n and the checkpoint is updated correspondingly.

        :param write_function: a FEniCS function being sent to the other participant as boundary condition at the coupling interface
        :param u_np1: new value of FEniCS solution u_n+1 at time t_n+1 = t+dt
        :param u_n: old value of FEniCS solution u_n at time t_n = t; updated via call-by-reference
        :param t: current time t_n for timestep n
        :param dt: timestep size dt = t_n+1 - t_n
        :param n: current timestep
        :return: return starting time t and timestep n for next FEniCS solver iteration. u_n is updated by advance correspondingly.
        """

        precice_step_complete = False
        fsi_converged=False

        # sample write data at interface
        if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
            #if T:self.init_T=float(T)
            #if dt:self.init_dt=float(dt)
            #self.init_T=float(T)
            #self.init_dt=float(dt)
            #if init_data1:self.init_data1=init_data1.copy(True)
            #if init_data2:self.init_data2=init_data2.copy(True)
            #if init_data3:self.init_data3=init_data3.copy(True)
            #fsi_converged=True
            self._interface.fulfilled_action(precice.action_write_iteration_checkpoint())

        self._write_data = self.convert_fenics_to_precice(write_function, self.write_local_dofs)

        # communication
        if self.read_n_vertices>0:
            if self._write_data_name=="Force" or self._write_data_name=="Displacement":
                self._interface.write_block_vector_data(self._write_data_id, self.write_n_vertices, self.write_vertex_ids, self._write_data)
            elif self._write_data_name=="Pressure":
                self._interface.write_block_scalar_data(self._write_data_id, self.write_n_vertices, self.write_vertex_ids, self._write_data)
        self._precice_tau = self._interface.advance(float(dt))
        dt.assign(min(self._precice_tau, float(dt)))
        if self.read_n_vertices>0:
            self._interface.read_block_vector_data(self._read_data_id, self.read_n_vertices, self.read_vertex_ids, self._read_data)

        read_function.vector()[self.read_local_dofs]=self._read_data

        # update boundary condition with read data
        #self._coupling_bc_expression.update_boundary_data(self._read_data, x_vert, y_vert)

#        precice_step_complete = False
#        fsi_converged=False

        # checkpointing
        if self._interface.is_action_required(precice.action_read_iteration_checkpoint()):#not yet converged
            #if T:self.init_T=float(T)
            #if dt:self.init_dt=float(dt)
            #if init_data1:init_data1.vector()[:]=self.init_data1.vector() 
            #if init_data2:init_data2.vector()[:]=self.init_data2.vector()
            #if init_data3:init_data3.vector()[:]=self.init_data3.vector()
            #dt.assign(self.init_dt)
            #T.assign(self.init_T)
            self._interface.fulfilled_action(precice.action_read_iteration_checkpoint())
        else:
            precice_step_complete=True

        #if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
            #if T:self.init_T=float(T)
            #if dt:self.init_dt=float(dt)
        #    self.init_T=float(T)
        #    self.init_dt=float(dt)
        #    self.init_data1=init_data1.copy(True)
        #    if init_data2:self.init_data2=init_data2.copy(True)
        #    if init_data3:self.init_data3=init_data3.copy(True)
        #    fsi_converged=True
        #    self._interface.fulfilled_action(precice.action_write_iteration_checkpoint())


        #if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
        #    self._interface.fulfilled_action(precice.action_write_iteration_checkpoint())

        return precice_step_complete, self._interface.is_timestep_complete()

    def initialize(self, coupling_subdomain, read_field, write_field, mapping=None, T=None, dt=None, init_data1=None, init_data2=None, init_data3=None):
        """Initializes remaining attributes. Called once, from the solver.

        :param read_field: function applied on the read field
        :param write_field: function applied on the write field
        """

        read_function_space=read_field.function_space()
        try: 
            write_function_space=write_field.function_space()
        except:
            write_function_space=read_function_space

        self.set_coupling_mesh(read_function_space, write_function_space, coupling_subdomain, mapping)
        self.set_write_field(write_field)
        self.set_read_field()
        self._precice_tau = self._interface.initialize()
        #dt.assign(min(self._precice_tau, float(dt)))

        if self._interface.is_action_required(precice.action_write_initial_data()):
            if self.write_n_vertices>0:
                if self._write_data_name=="Force" or self._write_data_name=="Displacement":
                    self._interface.write_block_vector_data(self._write_data_id, self.write_n_vertices, self.write_vertex_ids, self._write_data)
                elif self._write_data_name=="Pressure":
                    self._interface.write_block_scalar_data(self._write_data_id, self.write_n_vertices, self.write_vertex_ids, self._write_data)
            self._interface.fulfilled_action(precice.action_write_initial_data())

        self._interface.initialize_data()

        if self._interface.is_read_data_available():
            if self.read_n_vertices>0:
                self._interface.read_block_vector_data(self._read_data_id, self.read_n_vertices, self.read_vertex_ids, self._read_data)

        if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
            #if T:self.init_T=float(T)
            #if dt:self.init_dt=float(dt)
            #if init_data1:self.init_data1=init_data1.copy(True)
            #if init_data2:self.init_data2=init_data2.copy(True)
            #if init_data3:self.init_data3=init_data3.copy(True)
            self._interface.fulfilled_action(precice.action_write_iteration_checkpoint())
        
        #if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
        #    self.dt_init=float(dt)
        #    self._interface.fulfilled_action(precice.action_write_iteration_checkpoint())

        read_field.vector()[self.read_local_dofs]=self._read_data

        return self._precice_tau

    def is_coupling_ongoing(self):
        """Determines whether simulation should continue. Called from the
        simulation loop in the solver.

        :return: True if the coupling is ongoing, False otherwise
        """
        return self._interface.is_coupling_ongoing()

    def extract_coupling_boundary_coordinates(self):
        """Extracts the coordinates of vertices that lay on the boundary. 3D
        case currently handled as 2D.

        :return: x and y cooridinates.
        """
        vertices, _ = self.extract_coupling_boundary_vertices()
        vertices_x = vertices[0, :]
        vertices_y = vertices[1, :]
        if self._dimensions == 3:
            vertices_z = vertices[2, :]

        if self._dimensions == 2:
            return vertices_x, vertices_y
        elif self._dimensions == 3:
            # todo this has to be fixed for "proper" 3D coupling. Currently this is a workaround for the coupling of 2D fenics with pseudo 3D openfoam
            return vertices_x, vertices_y

    def finalize(self):
        """Finalizes the coupling interface."""
        self._interface.finalize()

    def get_solver_name(self):
        """Returns name of this solver as defined in config file.

        :return: Solver name.
        """
        return self._solver_name
