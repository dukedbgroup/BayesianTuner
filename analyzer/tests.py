# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import logging
import json
import os
import pandas as pd
from unittest import TestCase
from time import sleep
from pathlib import Path
from api_service.app import app as api_service_app
from api_service.db import metricdb, configdb
from analyzer.bayesian_optimizer_pool import BayesianOptimizerPool
from analyzer.bayesian_optimizer import get_candidate
from logger import get_logger

logger = get_logger(__name__, log_level=("TEST", "LOGLEVEL"))


class BayesianOptimizationTest(TestCase):

    def setUp(self):
        logger.debug('Creating flask clients')
        self.client = api_service_app.test_client()
        self.client2 = api_service_app.test_client()

    def getTestRequest(self):
        return {"appName": "redis",
                "data": [
                    {"instanceType": "t2.large",
                     "qosValue": 200.
                     },
                    {"instanceType": "m4.large",
                     "qosValue": 100.
                     }
                ]}

    # def testFlowSingleCient(self):
    #     fake_uuid = "8whe-weui-qjhf-38os"
    #     response = json.loads(self.client.post('/get-next-instance-types/' + fake_uuid,
    #                                            data=json.dumps(self.getTestRequest()),
    #                                            content_type="application/json").data)
    #     logger.debug(f"Response from posting request: {self.getTestRequest()}")
    #     logger.debug(response)

    #     while True:
    #         response = json.loads(self.client.get(
    #             "/get-optimizer-status/" + fake_uuid).data)
    #         logger.debug("Response after sending GET /get-optimizer-status")
    #         logger.debug(response)

    #         if response['Status'] == 'Running':
    #             logger.debug("Waiting for 5 sec")
    #             sleep(5)
    #         else:
    #             break

    def testGuessBestTrialsDirect(self):
        import numpy as np
        # dimension=7, nsamples=2
        result = get_candidate(np.array([[6, 9, 9, 0, 8, 0, 9], [
            9, 8, 8, 0, 8, 5, 8]]), np.array([0.8, 0.7]), [(0, 1)] * 7)
        logger.debug(result)

    def testSingleton(self):
        # TODO: Test if the singleton works in multiprocess
        pass


class PredictionTest(TestCase):

    def setUp(self):
        logger.debug('Creating flask client')
        self.client = api_service_app.test_client()
        self.test_collection = 'test-collection'

    def getTestRequest(self):
        return {'app1': 'testApp',
                'app2': 'testApp2',
                'model': 'LinearRegression1',
                'collection': self.test_collection}

    def testFlow(self):
        try:
            logger.debug(f'Getting database {metricdb.name}')
            db = metricdb._get_database()  # This triggers lazy-loading
            logger.debug('Setting up test documents')
            testFiles = (Path(__file__).parent /
                         'test_profiling_result').rglob('*.json')
            for path in testFiles:
                logger.debug("Adding: {}".format(path))
                with path.open('r') as f:
                    doc = json.load(f)
                    db[self.test_collection].insert_one(doc)
            response = self.client.post('/cross-app/predict',
                                        data=json.dumps(
                                            self.getTestRequest()),
                                        content_type="application/json")
            self.assertEqual(response.status_code, 200, response)
            data = json.loads(response.data)

            logger.debug('====Request====\n')
            logger.debug(self.getTestRequest())
            logger.debug('\n====Cross-App Interference Score Prediction====')
            logger.debug('\n' + str(pd.read_json(response.data)))
        except Exception as e:
            raise e
        finally:
            logger.debug(f'Clean up test collection: {self.test_collection}')
            db[self.test_collection].drop()
            db.client.close()
            logger.debug('Client connection closed')