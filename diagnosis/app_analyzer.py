import time
import requests
from collections import namedtuple
from math import isnan

from influxdb import InfluxDBClient

from diagnosis.derived_metrics import MetricsConsumer
from diagnosis.diagnosis import Diagnosis
from diagnosis.problems_detector import ProblemsDetector
from config import get_config

config = get_config()
WINDOW = int(config.get("ANALYZER", "CORRELATION_WINDOW_SECOND"))
INTERVAL = int(config.get("ANALYZER", "DIAGNOSIS_INTERVAL_SECOND"))
NANOSECONDS_PER_SECOND = 1000000000


class AppAnalyzer(object):
    def __init__(self, config):
        self.config = config
        self.metrics_consumer = MetricsConsumer(
            self.config.get("ANALYZER", "DERIVED_SL_CONFIG"),
            self.config.get("ANALYZER", "DERIVED_METRIC_CONFIG"))
        self.diagnosis = Diagnosis()
        self.problems_detector = ProblemsDetector(config)
        influx_host = config.get("INFLUXDB", "HOST")
        influx_port = config.get("INFLUXDB", "PORT")
        influx_db = config.get("INFLUXDB", "RESULT_DB_NAME")
        requests.post("http://%s:%s/query" % (influx_host, influx_port), params="q=CREATE DATABASE %s" % influx_db)
        self.influx_client = InfluxDBClient(
            influx_host,
            influx_port,
            config.get("INFLUXDB", "USER"),
            config.get("INFLUXDB", "PASSWORD"),
            influx_db)
        self.influx_client.create_retention_policy('result_policy', '2w', 1, default=True)

    def loop_all_app_metrics(self, start_time, batch_window, batch_interval):
        it = 1
        while True:
            end_time = start_time + batch_window
            print("\nIteration %d - Processing metrics from start: %d, to end: %d" %
                  (it, start_time, end_time))
            derived_metrics = self.metrics_consumer.get_derived_metrics(
                start_time, end_time)
            if derived_metrics.app_metrics is None:
                print("No app metrics found, exiting..")
                return
            metrics_with_cs = self.diagnosis.process_metrics(derived_metrics)
            self.write_results(metrics_with_cs, end_time)
            self.problems_detector.detect(metrics_with_cs)
            start_time += batch_interval
            it += 1

    def write_results(self, metrics, end_time):
        points_json = []
        for metric in metrics:
            point_json = {}
            # In Influx, measurements in two different databases cannot have the same name.
            # Below, we avoid a name conflict with derivedmetrics database.
            point_json["measurement"] = metric.metric_name + "_result"
            point_json["time"] = end_time
            fields = {}
            fields["average"] = float(metric.average)
            fields["correlation"] = float(metric.correlation)
            fields["confidence_score"] = float(metric.confidence_score)
            for field in ["average", "correlation", "confidence_score"]:
                if isnan(fields[field]):
                    fields[field] = None
            if not any(fields[field] for field in ["average", "correlation", "confidence_score"]):
                # InfluxDB requires non-empty data points.
                continue
            point_json["fields"] = fields
            tags = {}
            tags["resource_type"] = metric.resource_type
            tags["node_name"] = metric.node_name
            tags["pod_name"] = metric.pod_name
            point_json["tags"] = tags
            points_json.append(point_json)

        self.influx_client.write_points(points_json)


if __name__ == "__main__":
    aa = AppAnalyzer(config)
    aa.loop_all_app_metrics(1510967731000482000, WINDOW * NANOSECONDS_PER_SECOND, INTERVAL * NANOSECONDS_PER_SECOND)
