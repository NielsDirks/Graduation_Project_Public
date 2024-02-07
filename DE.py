import numpy as np
import power_grid_model.errors
from scipy._lib._util import check_random_state, MapWrapper
from scipy.optimize import OptimizeResult

from objective_parameters import OriginalParameters, ObjectiveParameters, OptimisedParameters
from pgm import PGM
from plotting import Plot


def differential_evolution(settings, measurement_idx, seed=None, workers=1):

    with DifferentialEvolutionSolver(settings,
                                     measurement_idx=measurement_idx,
                                     seed=seed,
                                     workers=workers) as solver:
        result = solver.solve()

    return result


class DifferentialEvolutionSolver:

    _binomial = {'best1bin': '_best1',
                 'rand1bin': '_rand1',
                 'randtobest1bin': '_randtobest1',
                 'currenttobest1bin': '_currenttobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2'}
    _exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'currenttobest1exp': '_currenttobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2'}

    def __init__(self, settings, measurement_idx, seed=None, workers=1):

        self.settings = settings
        self.measurement_idx = measurement_idx

        # read grid and set measurement data to loads and generators
        self.pgm = PGM(self)

        # get bounds and integrality from pgm
        bounds, integrality = self.pgm.get_bounds()

        # get the original PQ-point from a power flow with original measurement data and no reactive power control
        self.original_pfc = OriginalParameters(self)

        self.population = None
        self.population_energies = None

        if self.settings.de_strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[self.settings.de_strategy])
        elif self.settings.de_strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[self.settings.de_strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.de_strategy = self.settings.de_strategy

        self._mapwrapper = MapWrapper(workers)

        # Mutation constant should be in [0, 2). If specified as a sequence then dithering is performed.
        self.F = self.settings.mutation_const
        if not (np.all(np.isfinite(self.F)) or np.any(np.array(self.F) >= 2) or np.any(np.array(self.F) < 0)):
            raise ValueError('The mutation constant must be a float in U[0, 2), or specified as a tuple(min, max)'
                             ' where min < max and min, max are in U[0, 2).')

        self.dither = None
        if hasattr(self.F, '__iter__') and len(self.F) > 1:
            self.dither = [self.F[0], self.F[1]]
            self.dither.sort()

        self.CR = self.settings.crossover_const

        self.limits = np.array(bounds, dtype='float').T

        if np.size(self.limits, 0) != 2 or not np.all(np.isfinite(self.limits)):
            raise ValueError('bounds should be a sequence containing real valued (min, max) pairs for each value in x')

        self.maxiter = self.settings.maxiter
        if self.maxiter <= 0:
            raise ValueError('The maximum number of iterations must be larger than 0.')

        # maxfun can be implemented (which is not done in this algorithm). See scipy differential evolution algorithm
        self.maxfun = np.inf

        # population is scaled to between [0, 1]. We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])
        with np.errstate(divide='ignore'):
            # if lb == ub then the following line will be 1/0, which is why we ignore the divide by zero warning.
            # The result from 1/0 is inf, so replace those values by 0.
            self.__recip_scale_arg2 = 1 / self.__scale_arg2
            self.__recip_scale_arg2[~np.isfinite(self.__recip_scale_arg2)] = 0

        self.parameter_count = np.size(self.limits, 1)

        self.random_number_generator = check_random_state(seed)

        # Which parameters are going to be integers?
        if np.any(integrality):
            # user has provided a truth value for integer constraints
            integrality = np.broadcast_to(integrality, self.parameter_count)
            integrality = np.asarray(integrality, bool)
            # For integrality parameters change the limits to only allow integer values lying between the limits.
            lb, ub = np.copy(self.limits)

            lb = np.ceil(lb)
            ub = np.floor(ub)
            if not (lb[integrality] <= ub[integrality]).all():
                # there's a parameter that doesn't have an integer value lying between the limits
                raise ValueError("One of the integrality constraints does not have any possible integer values between"
                                 " the lower/upper bounds.")
            nlb = np.nextafter(lb[integrality] - 0.5, np.inf)
            nub = np.nextafter(ub[integrality] + 0.5, -np.inf)

            self.integrality = integrality
            self.limits[0, self.integrality] = nlb
            self.limits[1, self.integrality] = nub
        else:
            self.integrality = False

        # check for bounds where min and max are equal
        eb = self.limits[0] == self.limits[1]
        eb_count = np.count_nonzero(eb)

        # default population initialization is a latin hypercube design, but there are other population initializations
        # possible. The minimum is 5 because 'best2bin' requires a population that's at least 5 long
        # 202301 - reduced population size to account for parameters with equal bounds. If there are no varying
        # parameters set N to at least 1
        popsize = self.settings.popsize
        self.num_population_members = max(5, popsize * max(1, self.parameter_count - eb_count))
        self.population_shape = (self.num_population_members, self.parameter_count)

        self._nfev = 0
        if self.settings.pop_init_strategy == 'latinhypercube':
            self._init_population_lhs()
        elif self.settings.pop_init_strategy == 'random':
            self._init_population_random()
        else:
            raise ValueError("The population initialization method must be one of 'latinhypercube' or 'random'")

        self.disp = self.settings.disp_progress

        # initialise for plotting
        if self.disp:
            self.plot = Plot(self)

    def _init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Each parameter range needs to be sampled uniformly. The scaled parameter range ([0, 1)) needs to be split
        # into `self.num_population_members` segments, each of which has the following size:
        segsize = 1.0 / self.num_population_members

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (segsize * rng.uniform(size=self.population_shape)

                   # Offset each segment to cover the entire parameter range [0, 1)
                   + np.linspace(0., 1., self.num_population_members, endpoint=False)[:, np.newaxis])

        # Create an array for population of candidate solutions.
        self.population = np.zeros_like(samples)

        # Initialize population of candidate solutions by permutation of the random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # reset population energies
        self.population_energies = np.full(self.num_population_members, np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def _init_population_random(self):
        """
        Initializes the population at random. This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.uniform(size=self.population_shape)

        # reset population energies
        self.population_energies = np.full(self.num_population_members, np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    @property
    def x(self):
        """
        The best solution from the solver
        """
        return self._scale_parameters(self.population[0])

    def _converged(self):
        """
        Return True if the solver has converged.
        """
        if np.any(np.isinf(self.population_energies)):
            return False
        return np.std(self.population_energies) <= self.settings.tolerance * np.abs(np.mean(self.population_energies))

    def solve(self):
        nit, warning_flag = 0, False
        status_message = 'Optimization terminated successfully.'

        # do the optimisation
        for nit in range(1, self.maxiter + 1):
            # evolve the population by a generation
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                if self._nfev > self.maxfun:
                    status_message = 'Maximum number of function evaluations has been exceeded.'
                elif self._nfev == self.maxfun:
                    status_message = 'Maximum number of function evaluations has been reached.'
                break

            if self.disp:
                print("differential_evolution step %d: f(x) = %g, number of feasible solutions: %g"
                      % (nit, self.population_energies[0], np.count_nonzero(self.feasible)))
                # plot generation if figure exists
                if self.plot.fig:
                    self.plot.plot_generation(self, nit)

            # should the solver terminate?
            if warning_flag or self._converged():
                break

        # if for loop did not encounter the break statement
        else:
            status_message = 'Maximum number of iterations has been exceeded.'
            warning_flag = True

        # redo power flow calculation (pfc) with the best parameters, so pgm output data can be sent back to main.py
        optimised_pfc = OptimisedParameters(self)

        # for every DSO generator, get the reactive power in Mvar instead of the step of the DSO generator
        solution = self.x
        if self.pgm.n_dso_gens:
            solution[:self.pgm.n_dso_gens] = solution[:self.pgm.n_dso_gens] * self.settings.dso_gen_step_size

        differential_evolution_result = OptimizeResult(
            x=solution,
            fun=self.population_energies[0],
            nfev=self._nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is not True),
            vision_info=self.pgm.vision_info,
            date_time=self.pgm.measurement_info.date_time,
            original_pfc=self.original_pfc,
            optimised_pfc=optimised_pfc)

        differential_evolution_result.maxcv = np.max(self.constraint_violation)
        if differential_evolution_result.maxcv > 0:
            # if the result is infeasible then success must be False
            differential_evolution_result.success = False
            differential_evolution_result.message = f"The solution does not satisfy the constraints," \
                                                    f"MAXCV = {differential_evolution_result.maxcv}"

        # close optimisation plot
        if self.disp:
            self.plot.close_plot()

        return differential_evolution_result

    def _calculate_population_energies(self, population):
        """
        Calculate the energies of a population.

        Parameters
        ----------
        population : ndarray
            An array of parameter vectors normalised to [0, 1] using lower and upper limits.
            Has shape ``(np.size(population, 0), N)``.

        Returns
        -------
        energies : ndarray
            An array of energies corresponding to each population member. If maxfun will be exceeded during this call,
            then the number of function evaluations will be reduced and energies will be right-padded with np.inf.
            Has shape ``(np.size(population, 0),)``
        feasibilities : ndarray
            Boolean array of feasibility for each population member. Has shape ``(np.size(population, 0),)``
        constraint_violations : ndarray
            Total violation of constraints. Has shape ``(np.size(population, 0), M)``, where M is the total number of
            constraint components
        """
        num_members = np.size(population, 0)
        # s is the number of function evals left to stay under the maxfun budget
        s = int(min(float(num_members), self.maxfun - self._nfev))  # make num_members a float, otherwise warning

        parameters_pop = self._scale_parameters(population)
        try:
            # implement multiprocessing here: https://docs.python.org/3/library/multiprocessing.html
            # calc_energies = list(self._mapwrapper(self.func, parameters_pop[0:s]))
            # calc_energies = np.squeeze(calc_energies)
            calc_energies = [np.inf] * s
            calc_feasibilities = [False] * s
            calc_constraint_violations = [[np.inf]] * s
            for i in range(s):
                calc_energies[i], calc_feasibilities[i], calc_constraint_violations[i] = \
                    self._get_objective_value(parameters_pop[i])
            calc_energies = np.squeeze(calc_energies)
            calc_feasibilities = np.squeeze(calc_feasibilities)
            # do not squeeze calc_violations, as it is already in the right form
        except (TypeError, ValueError) as e:
            # wrong number of arguments for _mapwrapper or wrong length returned from the mapper
            raise RuntimeError("The map-like callable must be of the form f(func, iterable), returning a sequence of "
                               "numbers the same length as 'iterable'") from e

        if calc_energies.size != s:
            raise RuntimeError("func(x, *args) must return a scalar value")

        # first set all values to inf or False (worst values)
        energies = np.full(num_members, np.inf)
        feasibilities = np.full(num_members, False)
        # get number of constraints from the result in calc_constraint_violations
        # the number of constraints are set in the class ObjectiveParameters
        num_constraints = len(calc_constraint_violations[0])
        constraint_violations = np.full((num_members, num_constraints), np.inf)

        # then set the values for which calculations have been done
        energies[0:s] = calc_energies
        feasibilities[0:s] = calc_feasibilities
        constraint_violations[0:s] = calc_constraint_violations

        self._nfev += s

        return energies, feasibilities, constraint_violations

    def _promote_lowest_energy(self):
        # swaps 'best solution' into first population entry

        idx = np.arange(self.num_population_members)
        feasible_solutions = idx[self.feasible]
        if feasible_solutions.size:
            # find the best feasible solution
            idx_t = np.argmin(self.population_energies[feasible_solutions])
            h = feasible_solutions[idx_t]
        else:
            # no solution was feasible, use 'best' infeasible solution, which
            # will violate constraints the least
            h = np.argmin(np.sum(self.constraint_violation, axis=1))

        self.population_energies[[0, h]] = self.population_energies[[h, 0]]
        self.population[[0, h], :] = self.population[[h, 0], :]
        self.feasible[[0, h]] = self.feasible[[h, 0]]
        self.constraint_violation[[0, h], :] = self.constraint_violation[[h, 0], :]

    # noinspection PyMethodMayBeStatic
    def _accept_trial(self, energy_trial, feasible_trial, cv_trial, energy_orig, feasible_orig, cv_orig):
        """
        Trial is accepted if:
        * it satisfies all constraints and provides a lower or equal objective
          function value, while both the compared solutions are feasible
        - or -
        * it is feasible while the original solution is infeasible,
        - or -
        * it is infeasible, but provides a lower or equal constraint violation
          for all constraint functions.

        This test corresponds to section III of Lampinen [1]_.

        Parameters
        ----------
        energy_trial : float
            Energy of the trial solution
        feasible_trial : float
            Feasibility of trial solution
        cv_trial : array-like
            Excess constraint violation for the trial solution
        energy_orig : float
            Energy of the original solution
        feasible_orig : float
            Feasibility of original solution
        cv_orig : array-like
            Excess constraint violation for the original solution

        Returns
        -------
        accepted : bool

        """
        if feasible_orig and feasible_trial:
            return energy_trial <= energy_orig
        elif feasible_trial and not feasible_orig:
            return True
        elif not feasible_trial and (cv_trial <= cv_orig).all():
            # cv_trial < cv_orig would imply that both trial and orig are not feasible
            return True

        return False

    def __next__(self):
        """
        Evolve the population by a single generation

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # The population may have just been initialized (all entries are np.inf).
        # If it has you have to calculate the initial energies, feasibilities and constraint violation values
        if np.all(np.isinf(self.population_energies)):
            self.population_energies, self.feasible, self.constraint_violation = \
                self._calculate_population_energies(self.population)

            self._promote_lowest_energy()

            # plot generation 0 if figure exists
            try:
                if self.plot.fig:
                    self.plot.plot_generation(self, nit=0)
            except AttributeError:
                pass

        if self.dither is not None:
            self.F = self.random_number_generator.uniform(self.dither[0], self.dither[1])

        # Update the best solution immediately (for 'deferred' see scipy differential evolution algorithm)
        for candidate in range(self.num_population_members):
            if self._nfev > self.maxfun:
                raise StopIteration

            # create a trial solution
            trial = self._mutate(candidate)

            # ensuring that it's in the range [0, 1)
            self._ensure_constraint(trial)

            # scale from [0, 1) to the actual parameter value
            parameters = self._scale_parameters(trial)

            # determine the energy, feasibility and constraint_violation of the objective function
            energy, feasible, cv = self._get_objective_value(parameters)
            self._nfev += 1

            # compare trial and population member
            if self._accept_trial(energy, feasible, cv,
                                  self.population_energies[candidate],
                                  self.feasible[candidate],
                                  self.constraint_violation[candidate]):
                self.population[candidate] = trial
                self.population_energies[candidate] = np.squeeze(energy)
                self.feasible[candidate] = feasible
                self.constraint_violation[candidate] = cv

                # if the trial candidate is also better than the best solution then promote it.
                if self._accept_trial(energy, feasible, cv,
                                      self.population_energies[0],
                                      self.feasible[0],
                                      self.constraint_violation[0]):
                    self._promote_lowest_energy()

        return self.x, self.population_energies[0]

    def _scale_parameters(self, trial):
        """Scale from a number between 0 and 1 to parameters."""
        # trial either has shape (N, ) or (L, N), where L is the number of solutions being scaled
        scaled = self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2
        if np.any(self.integrality):
            i = np.broadcast_to(self.integrality, scaled.shape)
            scaled[i] = np.round(scaled[i])
        return scaled

    def _mutate(self, candidate):
        """Create a trial vector based on a mutation strategy."""
        trial = np.copy(self.population[candidate])

        rng = self.random_number_generator

        fill_point = rng.choice(self.parameter_count)

        if self.de_strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidate, self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        if self.de_strategy in self._binomial:
            crossovers = rng.uniform(size=self.parameter_count)
            crossovers = crossovers < self.CR
            # The last one is always from the bprime vector for binomial. If you fill in modulo with a loop you have
            # to set the last one to true. If you don't use a loop then you can have any random entry be True.
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        elif self.de_strategy in self._exponential:
            i = 0
            crossovers = rng.uniform(size=self.parameter_count)
            crossovers = crossovers < self.CR
            crossovers[0] = True
            while i < self.parameter_count and crossovers[i]:
                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1

            return trial

    def _ensure_constraint(self, trial):
        """Make sure the parameters lie between the limits."""
        mask = np.where((trial > 1) | (trial < 0))
        trial[mask] = self.random_number_generator.uniform(size=mask[0].shape)

    def _best1(self, samples):
        """best1bin, best1exp"""
        r0, r1 = samples[:2]
        return self.population[0] + self.F * (self.population[r0] - self.population[r1])

    def _rand1(self, samples):
        """rand1bin, rand1exp"""
        r0, r1, r2 = samples[:3]
        return self.population[r0] + self.F * (self.population[r1] - self.population[r2])

    def _randtobest1(self, samples):
        """randtobest1bin, randtobest1exp"""
        r0, r1, r2 = samples[:3]
        bprime = np.copy(self.population[r0])
        bprime += self.F * (self.population[0] - bprime)
        bprime += self.F * (self.population[r1] - self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        """currenttobest1bin, currenttobest1exp"""
        r0, r1 = samples[:2]
        bprime = (self.population[candidate] + self.F * (self.population[0] - self.population[candidate] +
                                                         self.population[r0] - self.population[r1]))
        return bprime

    def _best2(self, samples):
        """best2bin, best2exp"""
        r0, r1, r2, r3 = samples[:4]
        bprime = (self.population[0] + self.F * (self.population[r0] + self.population[r1] -
                                                 self.population[r2] - self.population[r3]))
        return bprime

    def _rand2(self, samples):
        """rand2bin, rand2exp"""
        r0, r1, r2, r3, r4 = samples
        bprime = (self.population[r0] + self.F * (self.population[r1] + self.population[r2] -
                                                  self.population[r3] - self.population[r4]))
        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement. You can't have the original candidate either.
        """
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs

    def _get_objective_value(self, parameters):
        self.pgm.update_rpc_generators(parameters)
        # do the power flow calculation
        try:
            pfc_output_data = self.pgm.model.calculate_power_flow(max_iterations=50)
            obj_param = ObjectiveParameters(True, self, pfc_output_data)
        except power_grid_model.errors.PowerGridError:
            obj_param = ObjectiveParameters(False)

        return obj_param.value, obj_param.feasible, obj_param.constraint_violation

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._mapwrapper.__exit__(*args)
