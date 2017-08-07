#!/usr/bin/env python3
from __future__ import division, print_function

import threading
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from logger import get_logger

from .bayesian_optimizer import get_candidate
from .util import (compute_cost, decode_nodetype, encode_nodetype,
                   get_all_nodetypes, get_budget, get_feature_bounds,
                   get_price, get_slo_type, get_slo_value)

pd.set_option('display.width', 1000)  # widen the display
np.set_printoptions(precision=3)
logger = get_logger(__name__, log_level=(
    "BAYESIAN_OPTIMIZER_POOL", "LOGLEVEL"))


class BayesianOptimizerPool():
    """ This class manages the training samples for each app_id,
        dispatches jobs to the worker pool, and tracks the status of each job.
    """

    # TODO: Thread safe?
    __singleton_lock = threading.Lock()
    __singleton_instance = None

    @classmethod
    def instance(cls):
        """ The singleton instance of BayesianOptimizerPool """
        if not cls.__singleton_instance:
            with cls.__singleton_lock:
                if not cls.__singleton_instance:
                    cls.__singleton_instance = cls()
        return cls.__singleton_instance

    def __init__(self):
        # Map for sharing data samples throughout the optimization session for each app_id
        self.sample_map = {}
        # Map for storing the future objects for python concurrent tasks
        self.future_map = {}
        # Pool of worker processes for handling jobs
        self.worker_pool = ProcessPoolExecutor(max_workers=16)
        # TODO: add a self.completion_map to track whether the optimization process is complete

    # TODO: Add workflow to handle the termination when the length of the sample_map exceeds the max_samples limit
    def get_candidates(self, app_id, request_body):
        """ The public method to asychronously start the jobs for generating candidates.
        Args:
            app_id(str): unique key to identify the optimization session for each app
            request_body(dict): the request body sent from the client of the analyzer service
        """

        # Generate initial samples if the input request body is empty
        # TODO: add a try-catch block to handle failure in the submit process
        if not request_body.get('data'):
            self.future_map[app_id] = []
            init_samples = BayesianOptimizerPool.generate_initial_samples()
            for i in init_samples:
                future = self.worker_pool.submit(encode_nodetype, i)
                self.future_map[app_id].append(future)
        # Else dispatch the optimizers
        else:
            # create datafame
            df = BayesianOptimizerPool.create_sample_dataframe(request_body)
            # update the shared sample map
            self.update_sample_map(app_id, df)
            logger.info(f"Training data:\n{self.sample_map}")
            # fetch the latest sample_map
            dataframe = self.sample_map[app_id]
            # create the training data to Bayesian optimizer
            training_data_list = [BayesianOptimizerPool.make_optimizer_training_data(
                dataframe, objective_type=o)
                for o in ['perf_over_cost',
                          'cost_given_perf_limit', 'perf_given_cost_limit']]

            self.future_map[app_id] = []
            feature_bounds = get_feature_bounds(normalized=True)
            for training_data in training_data_list:
                acq = 'cei' if training_data.has_constraint() else 'ei'
                logger.info(f"[{app_id}]Dispatching optimizer with acquisition function: {acq}:\n{training_data}")
                future = self.worker_pool.submit(
                    get_candidate,
                    training_data.feature_mat,
                    training_data.objective_arr,
                    feature_bounds,
                    acq=acq,
                    constraint_arr=training_data.constraint_arr,
                    constraint_upper=training_data.constraint_upper
                )
                self.future_map[app_id].append(future)

        return {"status": "submitted"}

    def get_status(self, app_id):
        """ The public method to get the running state of current optimization jobs for an app.
        """
        # TODO: check if complete, if so call best_candidates=compute_optimum() and return the best_candidates

        future_list = self.future_map.get(app_id)
        if future_list:
            if any([future.running() for future in future_list]):
                return {"status": "running"}
            elif any([future.cancelled() for future in future_list]):
                return {"status": "server_error", "error": "task cancelled"}
            elif all([future.done() for future in future_list]):
                try:
                    candidates = [future.result() for future in future_list]
                except Exception as e:
                    return {"status": "server_error",
                            "error": str(e)}
                else:
                    logger.info(f"Candidates: {candidates}")
                    return {"status": "done",
                            "data": self.filter_candidates(app_id, [decode_nodetype(c) for c in candidates])}
            else:
                logger.info(f"Futures in unexpected state: {future_list}")
                return {"status": "server_error",
                        "error": "unexpected task state"}
        else:
            return {"status": "bad_request",
                    "error": f"app_id {app_id} is not found"}

    def update_sample_map(self, app_id, df):
        with self.__singleton_lock:
            if self.sample_map.get(app_id) is not None:
                # check if incoming df contains duplicated nodetypes with existing sample_map
                intersection = set(df['nodetype']).intersection(
                    set(self.sample_map[app_id]['nodetype']))
                if intersection:
                    logger.warning(f"Duplicated sample was sent from the client")
                    logger.warning(f"input:\n{df}\nsample_map:\n{self.sample_map.get(app_id)}")
                    logger.warning(f"intersection: {intersection}")

            self.sample_map[app_id] = pd.concat(
                [self.sample_map.get(app_id), df])

    def filter_candidates(self, app_id, candidates, max_samples=15):
        """ This method determines if the recommended candidates should be returned to the client.
        Terminate conditions: 1. if number of samples evaluated exceeds max_samples or,
                              2. if all the candidates are duplicates within this run or with previous runs
        """
        # remove duplicates within this run
        candidates = list(set(candidates))

        if self.sample_map.get(app_id) is None:
            return candidates
        # TODO: move this termination check into get_candidates()
        elif len(self.sample_map.get(app_id)) > max_samples:
            return []
        else:
            # return candidates while removing duplicates with previous runs
            return [c for c in candidates if c not in self.sample_map.get(app_id)['nodetype'].values]

    # def filter_candidates(self, app_id, candidates, max_run=10):
    #     """ This method determine if candidates should be returned to client. # FORCING SELECT OTHER CANIDATES
    #     Terminate conditions: 1. if number of run exceed than max_run or,
    #                           2. if the recommended nodetype is duplicated within this run or with previous runs
    #     """
    #     # remove duplicates within this run
    #     candidates_ = list(set(candidates))

    #     if self.sample_map.get(app_id) is None:
    #         return candidates
    #     elif len(self.sample_map.get(app_id)) >= max_run:  # remove duplicates with previous run
    #         return []
    #     else:
    #         results = []
    #         for c in candidates:
    #             x = c
    #             while (x in self.sample_map.get(app_id)['nodetype'].values) or (x in results):
    #                 x = np.random.choice([i['name'] for i in get_all_nodetypes().values()])
    #             results.append(x)

    #         return results

    @staticmethod
    def generate_initial_samples(init_samples=3):
        """ This method randomly select a 'instanceFamily',
            and randomly select a 'nodetype' from that family repeatly
        """
        dataframe = pd.DataFrame.from_dict(list(get_all_nodetypes().values()))
        instance_families = list(set(dataframe['instanceFamily']))
        available = list(instance_families)
        # check if there is any instance's instanceFamily field is empty string
        assert '' not in instance_families, "instanceFamily shouldn't be empty"
        result = []

        logger.debug("Generating random initial samples")
        while init_samples > 0:
            family = np.random.choice(instance_families, 1)[0]
            if family in available:
                nodetype = np.random.choice(
                    dataframe[dataframe['instanceFamily'] == family]['name'], 1)[0]
                result.append(nodetype)
                init_samples -= 1
                available.remove(family)

            # reset available when all families are visited
            if not available:
                available = list(instance_families)

        logger.debug(f"initial samples:\n{result}")
        return result

    # @staticmethod
    # def generate_initial_samples(init_samples=3):
    #     dataframe = pd.DataFrame.from_dict(get_all_nodetypes()['data'])
    #     result = np.random.choice(dataframe['name'], init_samples)
    #     return result

    @staticmethod
    def make_optimizer_training_data(df, objective_type=None):
        """ Convert the objective and constraints such the optimizer can always:
            1. maximize objective function such that 2. constraints function < constraint
        """
        class BOTrainingData():
            """ Training data for bayesian optimizer
            """

            def __init__(self, objective_type, feature_mat, objective_arr,
                         constraint_arr=None, constraint_upper=None):
                self.objective_type = objective_type
                self.feature_mat = feature_mat
                self.objective_arr = objective_arr
                self.constraint_arr = constraint_arr
                self.constraint_upper = constraint_upper

            def __str__(self):
                result = ""
                result += f'Feature_mat:\n{self.feature_mat}\n'
                result += f'Objective_type: {self.objective_type}\n'
                result += f'Objective_arr: {self.objective_arr.values}\n'
                if self.constraint_arr is not None:
                    result += f'Constraint_arr: {self.constraint_arr.values}\n'
                if self.constraint_upper is not None:
                    result += f'Constraint_upper: {self.constraint_upper}\n'
                return result

            def has_constraint(self):
                """ See if this trainning data has constraint """
                return self.constraint_upper is not None

        implmentation = ['perf_over_cost',
                         'cost_given_perf_limit', 'perf_given_cost_limit']
        if objective_type not in implmentation:
            raise NotImplementedError(f'objective_type: {objective_type} is not implemented.')

        feature_mat = np.array(df['feature'].tolist())
        slo_type = df['slo_type'].iloc[0]
        budget = df['budget'].iloc[0]
        perf_constraint = df['slo_value'].iloc[0]
        perf_arr = df['qos_value']

        # Convert metric so we always try to maximize performance
        if slo_type == 'latency':
            perf_arr = 1. / df['qos_value']
            perf_constraint = 1. / perf_constraint
        elif slo_type == 'throughput':
            pass
        else:
            raise AssertionError(f'invalid slo type: {slo_type}')

        # Compute objectives
        perf_over_cost = (perf_arr / df['cost']
                          ).rename("performance over cost")

        if objective_type == 'perf_over_cost':
            return BOTrainingData(objective_type, feature_mat, perf_over_cost)
        elif objective_type == 'cost_given_perf_limit':
            return BOTrainingData(objective_type, feature_mat, -df['cost'], -perf_arr, -perf_constraint)
        elif objective_type == 'perf_given_cost_limit':
            return BOTrainingData(objective_type, feature_mat, perf_arr, df['cost'], budget)
        else:
            raise UserWarning("Unexpected error")

    @staticmethod
    def create_sample_dataframe(request_body):
        """ Convert request_body to dataframe of training samples.
        Args:
                request_body(dict): request body sent from workload profiler
        Returns:
                dfs(dataframe): sample data organized in dataframe
        """
        app_name = request_body['appName']
        slo_type = get_slo_type(app_name)
        assert slo_type in ['throughput', 'latency'],\
            f'slo type should be either throughput or latency, but got {slo_type}'

        dfs = []
        for data in request_body['data']:
            nodetype = data['instanceType']
            qos_value = data['qosValue']

            df = pd.DataFrame({'app_name': [app_name],
                               'qos_value': [qos_value],
                               'slo_type': [slo_type],
                               'nodetype': [nodetype],
                               'feature': [encode_nodetype(nodetype)],
                               'cost': [compute_cost(get_price(nodetype), slo_type, qos_value)],
                               'slo_value': [get_slo_value(app_name)],
                               'budget': [get_budget(app_name)]
                               })
            dfs.append(df)
        if not dfs:
            raise AssertionError(f'dataframe is not created\nrequest_body:\n{request_body}')
        return pd.concat(dfs)
