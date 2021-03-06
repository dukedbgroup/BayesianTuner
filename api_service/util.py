from functools import lru_cache

import bson
from flask.json import JSONEncoder
from flask import jsonify
from pymongo import ReturnDocument
from pymongo.cursor import Cursor
from werkzeug.routing import BaseConverter, ValidationError
import numpy as np
import pandas as pd
import state.apps as apps
from scipy.spatial.distance import euclidean

from api_service.db import configdb, metricdb
from config import get_config
from logger import get_logger

logger = get_logger(__name__, log_level=("ANALYZER", "LOGLEVEL"))

config = get_config()
nodetype_collection = config.get("ANALYZER", "NODETYPE_COLLECTION")
app_collection = config.get("ANALYZER", "APP_COLLECTION")
my_region = config.get("ANALYZER", "MY_REGION")
cost_type = 'time' #config.get("ANALYZER", "COST_TYPE")

DEFAULT_CLOCK_SPEED = 2.3
DEFAULT_NET_PERF = 'Low'
DEFAULT_IO_THPT = 125
DEFAULT_COST = {'time': 100, 'LinuxOnDemand': 2.1, 'LinuxReserved': 1.14,
                'WindowsOnDemand': 2.6, 'WindowsReserved': 1.35}

NETWORK_DICT = {'Very Low': 50, 'Low': 100, 'Low to Moderate': 300, 'Moderate': 500, 'High': 1000,
                '10 Gigabit': 10000, 'Up to 10 Gigabit': 10000, '20 Gigabit': 20000}


class JSONEncoderWithMongo(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Cursor):
            return list(obj)
        elif isinstance(obj, bson.ObjectId):
            return str(obj)
        return super().default(obj)


class ObjectIdConverter(BaseConverter):
    def to_python(self, value):
        if not bson.ObjectId.is_valid(value):
            raise ValidationError()
        else:
            return bson.ObjectId(value)

    def to_url(self, value):
        return str(value)


def error_response(error_message, code):
    response = jsonify(error=error_message)
    response.status_code = code
    return response


def ensure_document_found(document, **kwargs):
    if document is None:
        response = jsonify({'status': "bad_request",
                            'error': "Target document not found"})
        response.status_code = 404
        return response
    else:
        if kwargs:
            document = {
                new_key: document[original_key]
                for new_key, original_key in kwargs.items()
            }
        response = jsonify(data=document)
        response.status_code = 200
        return response

def update_and_return_doc(app_id, update_doc, unset=False):
    updated_doc = apps.update_and_get_app(app_id, update_doc, unset)
    if updated_doc:
        result = jsonify(data=updated_doc)
        result.status_code = 200
        return result
    return error_response("Could not update app {app_id}.", 404)


def ensure_application_updated(app_id, update_doc):
    if apps.update_app(app_id, update_doc):
        return jsonify(status=200)

    return util.response("Could not update app.", 404)


def shape_service_placement(deploy_json):
    result = {}
    result["name"] = deploy_json["name"]
    result["nodeMapping"] = []
    for task in deploy_json["kubernetes"]["taskDefinitions"]:
        if task["deployment"]["metadata"].get("namespace", "default") == "default":
            task["deployment"]["metadata"]["namespace"] = "default"
            for mapping in deploy_json["nodeMapping"]:
                if mapping["task"] == task["family"]:
                    result["nodeMapping"].append(mapping)

    nodes_in_default = set(map(lambda m: m["id"], result["nodeMapping"]))
    result["clusterDefinition"] = {
        "nodes": [node for node in deploy_json["clusterDefinition"]["nodes"]
                  if node["id"] in nodes_in_default]
    }
    return result


@lru_cache(maxsize=1)
def get_all_nodetypes(region=my_region):
    """ Get all nodetypes from the database and convert them into a map.
    """
    df = pd.read_csv('/home/mayuresh/analyzer/perf_data/kmeans-exhaustive.tsv', sep='\t', header=0, index_col=0)
    # nodes_list = [1, 2, 3, 4]
    # cores_list_1 = [2, 4, 6, 8]
    # cores_list_2 = [1, 2, 3, 4]
    # cores_list_3 = [1, 2]
    # memory_list = [0.2, 0.4, 0.6, 0.8]
    # newratio_list = [1, 3, 5, 7]

    # print(df.head(10))
    nodetype_list = []
    for row in df.iterrows():
      index, data = row
      # print(index, ': ', data)
      nodetype_list.append({'name':index, 'nodes':data['numContainers'], 'cores':data['concurrency']*data['numContainers'], 'memory':data['cache'], 'newRatio':data['newratio'], 'instanceType':index, 'cost': {'time':data['time'], 'variance':data['sigma'], 'whitebox':data['U']}, 'extra': {'feature1':data['feature1'], 'feature2':data['feature2'], 'feature3':data['feature3']}})

    '''
    nodeId = 0
    for nodes in nodes_list:
      cores_list = cores_list_3
      if nodes == 1
        cores_list = cores_list_1
      if nodes == 2
        cores_list = cores_list_2
      for cores in cores_list
        for memory in memory_list:
          for newratio in newratio_list:
            nodetype_list.append({'name':nodeId, 
'nodes':nodes, 'cores':cores*nodes, 'memory':memory, 'newRatio':newratio,
'instanceType':nodeId, 'cost': {'time':time}})
            nodeId = nodeId + 1
    else:
      print('Enumerated nodetypes #', nodeId)
    '''
    '''
    region_filter = {'region': region}
    nodetype_list = configdb[nodetype_collection].find_one(region_filter)
    '''
    nodetype_map = {}
    if nodetype_list is None:
        raise KeyError(
            'Cannot find nodetype document: filter={}'.format(region_filter))
    
    # creating dictionary for nodetypes
    for nodetype in nodetype_list:
        nodetype_map[nodetype['name']] = nodetype

    return nodetype_map


def get_feature_bounds(normalized=False):
    nodetype_map = get_all_nodetypes()

    if normalized:
        features = [encode_nodetype(k) for k in nodetype_map]
    else:
        features = [get_raw_features(k) for k in nodetype_map]

    return get_bounds(features)


def get_bounds(vectors):
    """ Get the (min, max) bounds for each dimension of vector.
    """
    vectors = np.array(vectors)
    return list(zip(vectors.min(axis=0), vectors.max(axis=0)))


@lru_cache(maxsize=16)
def get_raw_features(nodetype_name):
    """ for each instance type, get a vector of raw feature values from the database
        TODO: improve query efficiency by precomputing & caching all feature vectors
    """
    nodetype_map = get_all_nodetypes()
    nodetype = nodetype_map.get(nodetype_name)
    if nodetype is None:
        raise KeyError(
            'Cannot find instance type in the database: name={nodetype_name}')
    nodes = nodetype['nodes']
    cores = nodetype['cores']
    memory = nodetype['memory']
    newRatio = nodetype['newRatio']
    #whitebox = nodetype['cost']['whitebox']
    feature_vector = np.array([nodes, cores, memory, newRatio])
    '''
    vcpu = nodetype['cpuConfig']['vCPU']
    clock_speed = nodetype['cpuConfig']['clockSpeed']['value']
    mem_size = nodetype['memoryConfig']['size']['value']
    net_perf = nodetype['networkConfig']['performance']
    try:
        io_thpt = nodetype['storageConfig']['expectedThroughput']['value']
    except KeyError:
        io_thpt = DEFAULT_IO_THPT

    if clock_speed == 0:
        clock_speed = DEFAULT_CLOCK_SPEED

    if net_perf == "":
        net_perf = DEFAULT_NET_PERF
    net_bw = NETWORK_DICT[net_perf]

    feature_vector = np.array(
        [vcpu, clock_speed, mem_size, net_bw, io_thpt])
    '''
    return feature_vector


@lru_cache(maxsize=16)
def get_extra_features(nodetype_name):
    nodetype_map = get_all_nodetypes()
    nodetype = nodetype_map.get(nodetype_name)
    if nodetype is None:
        raise KeyError(
            'Cannot find instance type in the database: name={nodetype_name}')
    f1 = nodetype['extra']['feature1']
    f2 = nodetype['extra']['feature2']
    f3 = nodetype['extra']['feature3']
    feature_vector = np.array([f1, f2, f3])
    return feature_vector


def extra_features_bounds():
    nodetype_map = get_all_nodetypes()
    features = [get_extra_features(k) for k in nodetype_map]
    return get_bounds(features)


@lru_cache(maxsize=16)
def encode_extra_nodetype(nodetype):
    """ convert each node type to a normalized feature vector
    """
    raw_features = get_extra_features(nodetype)
    raw_feature_bounds = np.array(extra_features_bounds())
    # normalizing each feature variable by its maximum value
    normalized_feature = np.divide(raw_features, raw_feature_bounds[:, 1])
    # logger.debug(f"Encoded feature vector for {nodetype}: {features_normalized}")

    return normalized_feature


@lru_cache(maxsize=16)
def encode_nodetype(nodetype):
    """ convert each node type to a normalized feature vector
    """
    raw_features = get_raw_features(nodetype)
    raw_feature_bounds = np.array(get_feature_bounds(normalized=False))
    # normalizing each feature variable by its maximum value
    normalized_feature = np.divide(raw_features, raw_feature_bounds[:, 1])
    # logger.debug(f"Encoded feature vector for {nodetype}: {features_normalized}")

    return normalized_feature


def decode_nodetype(feature_vector, available_nodetypes):
    """ convert a candidate solution recommended by the optimizer into a list VM node type sorted by distance
        Args:
            feature_vector: candidate solution in a vector space
        Returns:
            candidate_rank (list of tuples): sorted list of (euclidean distance, available nodetypes)
                i.e. [(0.353, 'nodetype1'), (0.526, 'nodetype2), ...]
    """
    return sorted([(euclidean(encode_nodetype(nodetype), feature_vector), nodetype)
                   for nodetype in available_nodetypes])


# get from configdb the price (hourly cost) of an nodetype
def get_price(nodetype_name):
    nodetype_map = get_all_nodetypes()
    nodetype = nodetype_map.get(nodetype_name)

    if nodetype is None:
        raise KeyError(
            f'Cannot find instance type in the database: name={nodetype_name}')

    try:
        price = nodetype['cost'][cost_type]
#        price = nodetype['hourlyCost'][cost_type]['value']
        if price == 0:
            price = DEFAULT_COST[cost_type]
    except KeyError:
        price = DEFAULT_COST[cost_type]

    return price


# cost function based on sloMetric type and value, and hourly price
def compute_cost(price, slo_type, qos_value):
    if slo_type in ['latency', 'throughput']:
        # for long-running services, calculate the montly cost based on hourly price
        cost = price * 24 * 30
    else:
        # for batch jobs, qos_value = job_completion_time in minutes
        cost = price * qos_value / 60

    return cost


# get from configdb the type of an application ("long-running" or "batch")
def get_app_type(app_name):
    return "long-running"
'''
    app = get_app_info(app_name)

    return app['type']
'''

# get from configdb the slo metric type of an application
def get_slo_type(app_name):
    return "latency"
'''
    app = get_app_info(app_name)

    return app['slo']['type']
'''

def get_slo_value(app_name):
    return 500.
'''
    app = get_app_info(app_name)

    try:
        slo_value = app['slo']['value']
    except KeyError:
        slo_value = 500.  # TODO: put an nan

    return slo_value
'''

def get_budget(app_name):
    return 25000.
'''
    app = get_app_info(app_name)

    try:
        budget = app['budget']['value']
    except KeyError:
        budget = 25000.  # all types allowed

    return budget
'''


# get from configdb the {cpu|mem} requests (i.e. min) for the app service container
def get_resource_requests(app_name):
    app = get_app_info(app_name)

    # TODO: handle apps with multiple services
    service = app['serviceNames'][0]

    min_resources = {'cpu': 0., 'mem': 0.}

    for task in app['taskDefinitions']:
        if task['nodeMapping']['task'] == service:
            container_spec = \
                task['taskDefinition']['deployment']['spec']['template']['spec']['containers'][0]
            try:
                resource_requests = container_spec['resources']['requests']
            except KeyError:
                logger.debug(
                    "No resource requests for the container running {service}")
                return min_resources

            try:
                cpu_request = resource_requests['cpu']
                # conver cpu unit from millicores to number of vcpus
                if cpu_request[-1] == 'm':
                    min_resources['cpu'] = float(
                        cpu_request[:len(cpu_request) - 1]) / 1000.
                else:
                    min_resources['cpu'] = float(cpu_request)
            except KeyError:
                logger.debug(
                    "No cpu request for the container running {service}")

            try:
                mem_request = resource_requests['memory']
                # convert memory unit to GB
                if mem_request[-1] == 'M':
                    min_resources['mem'] = float(
                        mem_request[:len(mem_request) - 1]) / 1000.
                elif mem_request[-2:] == 'Mi':
                    min_resources['mem'] = float(
                        mem_request[:len(mem_request) - 2]) / 1024.
                else:
                    min_resources['mem'] = float(mem_request) / 1024. / 1024.
            except KeyError:
                logger.debug(
                    "No memory request for the container running {service}")

    logger.info(
        "Found resource requests for app {app_name} service {service}: {min_resources}")
    return min_resources


# TODO: probably need to change to appId.
def create_profiling_dataframe(app_name, collection='profiling'):
    """ Create a dataframe of features.
    Args:
        app_name(str): Map to the 'appName' in Mongo database.
    Returns:
        df(pandas dataframe): Dataframe with rows of features, where each row is a service.
        (i.e. if an app has N services, where each service has K dimensions, the dataframe would be NxK)
    """
    filt = {'appName': app_name}
    app = metricdb[collection].find_one(filt)
    if app is None:
        raise KeyError(
            'Cannot find document: filter={}'.format(filt))
    serviceNames = pd.Index(app['services'])
    benchmarkNames = pd.Index(app['benchmarks'])

    # make dataframe
    ibenchScores = []
    for service in serviceNames:
        filt = {'appName': app_name, 'serviceInTest': service}
        app = metricdb[collection].find_one(filt)
        if app is None:
            raise KeyError(
                'Cannot find document: filter={}'.format(filt))
        ibenchScores.append([i['toleratedInterference']
                             for i in app['testResult']])
    df = pd.DataFrame(data=np.array(ibenchScores),
                      index=serviceNames, columns=benchmarkNames)
    return df


def get_calibration_dataframe(calibration_document):
    if calibration_document is None:
        return None
    df = pd.DataFrame(calibration_document["testResult"])
    df = df.rename(columns={"qosMetric": "qosValue"})
    data = df.groupby("loadIntensity").apply(
        lambda group: group["qosValue"].describe()[["mean", "min", "max"]]
    ).reset_index()
    return data.to_dict(orient="records")


def percentile(n):
    def f(series): return series.quantile(n / 100)

    f.__name__ = "percentile_{n}"
    return f


def get_profiling_dataframe(profiling_document):
    if profiling_document is None:
        return None
    df = pd.DataFrame(profiling_document["testResult"])
    df = df.rename(columns={"qos": "qosValue"})
    data = df.groupby("benchmark").apply(
        lambda group: group.groupby("intensity")["qosValue"].agg(
            ["mean", percentile(10), percentile(90)]
        ).reset_index().to_dict(orient="records")
    )
    return data.to_dict()


def get_radar_dataframe(profiling_document):
    # Currently calibration document and profilng document are not related
    # together.
    app_name = profiling_document['appName']
    slo = configdb.applications.find_one({'name': app_name})['slo']

    test_results = pd.DataFrame(profiling_document['testResult'])
    test_results = test_results.rename(columns={'qos': 'qosValue'})

    radar_data = {
        'benchmark': [],
        'tolerated_interference': [],
        'score': [],
    }

    for benchmark, df in test_results.groupby(['benchmark', 'intensity'])['qosValue'].mean().groupby(level='benchmark'):
        df.index = df.index.droplevel('benchmark')
        result = compute_tolerated_interference(
            benchmarks=df,
            slo_value=slo['value'],
            metric_type=slo['type'],
        )
        radar_data['benchmark'].append(benchmark)
        radar_data['tolerated_interference'].append(min(result))
        # compute score
        radar_data['score'].append(100. - min(result))

    return radar_data


def compute_tolerated_interference(benchmarks, slo_value, metric_type, tolerated_percentage=10.):
    """ Compute Tolerated Interference, with linear interpolation.
    Args:
        benchmarks(DataFrame): index=intensity and columns=['qosValue']
        slo_value(float): service level objective for the application
        metric_type(str): 'throughput', 'latency'
        tolerated_percentage(float): percentage of slo tolerance
    Return:
        ti(float): tolerated interference between [0, 100].
    """

    def _linearIntp(tup1, tup2, y3):
        x1, y1 = tup1
        x2, y2 = tup2
        if y1 > y2:
            return _linearIntp((x2, y2), (x1, y1), y3)
        if y3 < y1 or y3 > y2:
            return None
        else:
            return (y3 - y1) * (x2 - x1) / (y2 - y1) + x1

    intensities, slo_values = np.append(
        0, benchmarks.index.values), np.append(slo_value, benchmarks.values)
    candidates = []

    # check metric type
    if metric_type == 'throughput':
        tolerated_slo_value = slo_value * (1. - tolerated_percentage / 100.)
        if min(slo_values) > tolerated_slo_value:
            candidates.append(100.)
    elif metric_type == 'latency':
        tolerated_slo_value = slo_value * (1. + tolerated_percentage / 100.)
        if max(slo_values) < tolerated_slo_value:
            candidates.append(100.)
    else:
        assert False, 'invalid metric type'

    # check input data
    assert (sorted(intensities) == intensities).all(
    ), 'intensities are not monotonic. intensities: {}'.format(intensities)
    assert all(intensity >= 0 and intensity <=
               100 for intensity in intensities), 'invalid intensities. intensites: {}'.format(intensities)
    assert len(slo_values) == len(intensities), \
        'length of slo_values and intensities does not match. slo_values: {}, intensities: {}'.format(
            slo_values, intensities)

    for i in range(len(slo_values) - 1):  # edge case tested
        x = _linearIntp((intensities[i], slo_values[i]), (intensities[
                                                              i + 1], slo_values[i + 1]), tolerated_slo_value)
        if x:
            candidates.append(x)

    return sorted(candidates)
