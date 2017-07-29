#!/usr/bin/env python3
from __future__ import division, print_function

import threading
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from logger import get_logger

from .bayesian_optimizer import get_candidate
from .util import (compute_cost, decode_instance_type, encode_instance_type,
                   get_all_nodetypes, get_bounds, get_price, get_slo_type, get_slo_value, get_budget)

logger = get_logger(__name__, log_level=("BAYESIAN_OPTIMIZER", "LOGLEVEL"))


class BayesianOptimizerPool():
    """ This class manages the training samples for each app_id,
        dispatches the optimization jobs, and track the status of jobs.
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
        # Place for storing samples of each app_id.
        self.sample_map = {}
        # Place for storing python concurrent object.
        self.future_map = {}
        # Pool of worker processes
        self.worker_pool = ProcessPoolExecutor(max_workers=16)

    def get_candidates(self, app_id, request_body):
        """ The public method to dispatch optimizers asychronously.
        Args:
            app_id(str): unique key to identify the application
            request_body(dict): the request body sent from workload profiler
        Returns:
            None. client are expect to pull the results with get_status method.
        """
        # update the shared sample map
        updated = self.update_sample_map(app_id, request_body)
        # if nothing was updated, initialize point for workload profiler
        if not updated:
            print("Not Updated")
            self.future_map[app_id] = []
            future = self.worker_pool.submit(
                BayesianOptimizerPool.generate_initial_points)
            self.future_map[app_id].append(future)
            return

        # fetch the latest sample_map
        dataframe = self.sample_map[app_id]
        # create the training data to Bayesian optimizer
        training_data_list = [BayesianOptimizerPool.
                              make_optimizer_training_data(
                                  dataframe, objective_type=o)
                              for o in ['perf_over_cost',
                                        'cost_given_perf', 'perf_given_cost']]

        feature_bounds = get_bounds()
        self.future_map[app_id] = []

        for training_data in training_data_list:
            logger.info(f"[{app_id}]Dispatching optimizer:\n{training_data}")
            acq = 'cei' if training_data.has_constraint() else 'ei'

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

    def get_status(self, app_id):
        """ The public method to get the running state of each worker.
        """
        future_list = self.future_map.get(app_id)
        if future_list:
            if any([future.running() for future in future_list]):
                return {"Status": "Running"}
            elif any([future.cancelled() for future in future_list]):
                return {"Status": "Cancelled"}
            elif all([future.done() for future in future_list]):
                try:
                    candidates = [future.result() for future in future_list]
                except Exception as e:
                    return {"Status": "Exception",
                            "Data": str(e)}
                else:
                    return {"Status": "Done",
                            "Data": [decode_instance_type(c) for c in candidates
                                     if not self.should_terminate(decode_instance_type(c))]}
                return {"Status": "Unexpected future state"}
        else:
            return {"Status": "Not running"}

    def update_sample_map(self, app_id, request_body):
        # TODO: thread safe?
        # TODO: check if workload profiler sends duplicate samples.
        df = BayesianOptimizerPool.create_sample_dataframe(request_body)

        if df is not None:
            # if duplicated":
            #     raise AssertionError(
            #         'Duplicated sample was sent from workload profiler')
            # logger.warning(f"request body dump: \n{request_body}")
            self.sample_map[app_id] = pd.concat(
                [df, self.sample_map.get(app_id)])
            return True
        else:
            logger.warning(
                "empty dataframe was generated from request body")
            return False

    # TODO: implement this
    def should_terminate(self, instance_type):
        return False

    # TODO: implement this
    @staticmethod
    def generate_initial_points():
        return ['t2.large', 'p2.8xlarge', 'x1.32xlarge']

    @staticmethod
    def make_optimizer_training_data(df, objective_type=None):
        """ Convert the objective and constraints such the optimizer can always‰:
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
                return f'objective_type:\n{self.objective_type}\n' +\
                    f'feature_mat:\n{self.feature_mat}\n' +\
                    f'objective_arr:\n{self.objective_arr}\n' +\
                    f'constraint_arr:\n{self.constraint_arr}\n' +\
                    f'constraint_upper:\n{self.constraint_upper}\n'

            def has_constraint(self):
                """ See if this trainning data has constraint """
                return self.constraint_upper is not None

        implmentation = ['perf_over_cost',
                         'cost_given_perf', 'perf_given_cost']
        if objective_type not in implmentation:
            raise NotImplementedError(f'objective_type: {objective_type} is not implemented.')

        feature_mat = np.array(df['feature'].tolist())
        slo_type = df['slo_type'].iloc[0]
        budget = df['budget'].iloc[0]
        perf_constraint = df['slo'].iloc[0]
        perf_arr = df['qos_value']

        # Convert metric so we always try to maximize performance
        if slo_type == 'latency':
            perf_arr = 1. / df['qos_value']
            perf_constraint = 1 / perf_constraint
        elif slo_type == 'throughput':
            pass
        else:
            raise AssertionError(f'invalid slo type: {slo_type}')

        if objective_type == 'perf_over_cost':
            return BOTrainingData(objective_type, feature_mat, perf_arr / df['cost'])
        elif objective_type == 'cost_given_perf':
            return BOTrainingData(objective_type, feature_mat, df['cost'], -perf_arr, -perf_constraint)
        elif objective_type == 'perf_given_cost':
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
            instance_type = data['instanceType']
            qos_value = data['qosValue']

            df = pd.DataFrame({'app_name': [app_name],
                               'qos_value': [qos_value],
                               'slo_type': [slo_type],
                               'instance_type': [instance_type],
                               'feature': [encode_instance_type(instance_type)],
                               'cost': [compute_cost(get_price(instance_type), slo_type, qos_value)],
                               'slo': [get_slo_value(app_name)],
                               'budget': [get_budget(app_name)]
                               })
            dfs.append(df)
        return pd.concat(dfs)
