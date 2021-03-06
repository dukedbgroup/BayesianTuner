"""
Harness for the analyzer logic. It only tests the quality of its predictions.
It mocks all other parts of the system.

Assumptions
- performance is throughput (higher is better)

TODO
- initialize analyzer
"""

__author__ = "Christos Kozyrakis"
__email__ = "christos@hyperpilot.io"
__copyright__ = "Copyright 2017, HyperPilot Inc"

from random import randint
import argparse
import json
import numpy as np
import sys, os
import time
import operator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '.'))
from api_service import util
import bayesian_optimizer_pool

# a couple of globals
features = {}
times = {}
variance = {}
cloud = None

class AwsPerf(object):
  """ A class for real performance data collected from AWS
  """

  def __init__(self, perf_file):
    """ initialize perf map from the result data json file
    """
    self.perf_file = perf_file
    with open(perf_file) as json_data:
        data = json.load(json_data)
    self.perf_map = data['testResult']


class CloudPerf(object):
  """ A class for a cloud performance model
  """

  def __init__(self,
               vcpu_a, vcpu_b, vcpu_c, vcpu_w, vcpu_s,
               clk_a, clk_b, clk_c, clk_w, clk_s,
               mem_a, mem_b, mem_c, mem_w, mem_s,
               net_a, net_b, net_c, net_w, net_s,
               # io_a, io_b, io_c, io_w, io_s,
               base, noise, nrange):
    """ initialize key parameters
    """
    self.vcpu_a = vcpu_a
    self.vcpu_b = vcpu_b
    self.vcpu_c = vcpu_c
    self.vcpu_s = vcpu_s
    self.clk_a = clk_a
    self.clk_b = clk_b
    self.clk_c = clk_c
    self.clk_s = clk_s
    self.mem_a = mem_a
    self.mem_b = mem_b
    self.mem_c = mem_c
    self.mem_s = mem_s
    self.net_a = net_a
    self.net_b = net_b
    self.net_c = net_c
    self.net_s = net_s
    # self.io_a = io_a
    # self.io_b = io_b
    # self.io_c = io_c
    # self.io_s = io_s
    self.vcpu_w = vcpu_w
    self.clk_w = clk_w
    self.mem_w = mem_w
    self.net_w = net_w
    # self.io_w = io_w
    self.noise = noise
    self.nrange = nrange
    self.base = base

    # some sanity checks
    if (vcpu_w + clk_w + mem_w + net_w) != 1.0:
      print("ERROR: Performance weights should sum to 1.0")
      sys.exit()
    if clk_a == 0 and clk_b == 0 and clk_c == 0 and clk_w != 0:
      print("ERROR: Clk performance parameters are all 0")
      sys.exit()
    if mem_a == 0 and mem_b == 0 and mem_c == 0 and mem_w != 0:
      print("ERROR: Mem performance parameters are all 0")
      sys.exit()
    if net_a == 0 and net_b == 0 and net_c == 0 and net_w != 0:
      print("ERROR: Net performance parameters are all 0")
      sys.exit()
    # if io_a == 0 and io_b == 0 and io_c == 0 and io_w != 0:
    #  print("ERROR: IO performance parameters are all 0")
    #  sys.exit()

  def vcpu_model(self, vcpu):
    """ a vcpu based performance model.
        perf = vcpu / (a + vcpu*b + vcpu^2*b)
        interesting cases:  b=c=0, b>0/c=0, b>>0/c=0, b>>0/c>0
        assumes vcpu_min = 1
    """
    base = vcpu * self.vcpu_s
    return base / (self.vcpu_a + self.vcpu_b*base + self.vcpu_c*base*base)

  def clk_model(self, clk):
    """ a clk based performance model.
        perf = clk / (a + clk*b + clk^2*b)
        assumes clk_min = 2.3
    """
    base = clk * self.clk_s
    return base / (self.clk_a + self.clk_b*base + self.clk_c*base*base)

  def mem_model(self, mem):
    """ a mem based performance model.
        perf = mem / (a + mem*b + mem^2*b)
        interesting cases:  b=c=0, b>0/c=0, b>>0/c=0, b>>0/c>0
        assumes mem_min = 0.5
    """
    base = mem * self.mem_s
    return base / (self.mem_a + self.mem_b*base + self.mem_c*base*base)

  def net_model(self, net):
    """ a net based performance model.
        perf = net / (a + net*b + net^2*b)
        interesting cases:  b=c=0, b>0/c=0, b>>0/c=0, b>>0/c>0
        assumes net_min = 100
    """
    base = net * self.net_s
    return base / (self.net_a + self.net_b*base + self.net_c*base*base)

  def io_model(self, io):
    """ a io based performance model.
        perf = io / (a + io*b + io^2*b)
        interesting cases:  b=c=0, b>0/c=0, b>>0/c=0, b>>0/c>0
        assumes io_min = 50
    """
    base = io * self.io_s
    return base / (self.io_a + self.io_b*base + self.io_c*base*base)

  def perf(self, vcpu, clk, mem, net, noise):
    """ estimate performance
    """
    perf = self.vcpu_w * self.vcpu_model(vcpu) + \
           self.clk_w * self.clk_model(clk) + \
           self.mem_w * self.mem_model(mem) + \
           self.net_w * self.net_model(net)
    #       self.io_w * self.io_model(io)
    if noise and self.noise:
      change = randint((-1) * self.nrange, self.nrange)
      perf += perf*change/100
    return self.base * perf


def str2bool(v):
  """ estimate performance
  """
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def nodeinfo(nodetype):
  """ returns a string with all the node info
  """
  global features
  global cloud
  global model
  if nodetype == "none":
    return "perf %10.2f, price %5.3f, cost %7.2f, perf/cost %10.2f" \
         %(0.0, 0.0, 0.0, 0.0)
  feat = features[nodetype]
  if model == 'aws': # use real data from aws
    perf = aws_data.perf_map[nodetype]['qosValue']
  else: # use CloudPerf model instead
    perf = cloud.perf(feat[0], feat[1], feat[2], feat[3], False)
  price = util.get_price(nodetype)
  cost = util.compute_cost(price, 'throughput', perf)
  return "perf %10.2f, price %5.3f, cost %7.2f, perf/cost %10.2f" \
         %(perf, price, cost, perf/cost)


def __main__():
  """ Main function of analyzer harness
  """
  # command line summary
  print("Running: ", ' '.join(map(str, sys.argv)))
  print()
  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--verbose", type=str2bool, required=False, default=False, help="increase output verbosity")
  parser.add_argument("-i", "--iter", type=int, required=False, default=50, help="maximum iterations")
  parser.add_argument("-n", "--noise", type=str2bool, required=False, default=True, help="add noise to cloud performance")
  parser.add_argument("-r", "--nrange", type=int, required=False, default=10, help="noise range (int)")
  parser.add_argument("-m", "--model", type=str, required=False, default="not-aws", help="perf model used (aws or cloud)")
  parser.add_argument("-va", "-vcpua", type=float, required=False, default=1.0, help="vcpu a")
  parser.add_argument("-vb", "-vcpub", type=float, required=False, default=0.0, help="vcpu b")
  parser.add_argument("-vc", "-vcpuc", type=float, required=False, default=0.0, help="vcpu c")
  parser.add_argument("-ca", "-clka", type=float, required=False, default=1.0, help="clk a")
  parser.add_argument("-cb", "-clkb", type=float, required=False, default=0.0, help="clk b")
  parser.add_argument("-cc", "-clkc", type=float, required=False, default=0.0, help="clk c")
  parser.add_argument("-ma", "-mema", type=float, required=False, default=1.0, help="mem a")
  parser.add_argument("-mb", "-memb", type=float, required=False, default=0.0, help="mem b")
  parser.add_argument("-mc", "-memc", type=float, required=False, default=0.0, help="mem c")
  parser.add_argument("-na", "-neta", type=float, required=False, default=1.0, help="net a")
  parser.add_argument("-nb", "-netb", type=float, required=False, default=0.0, help="net b")
  parser.add_argument("-nc", "-netc", type=float, required=False, default=0.0, help="net c")
  parser.add_argument("-ia", "-ioa", type=float, required=False, default=1.0, help="io a")
  parser.add_argument("-ib", "-iob", type=float, required=False, default=0.0, help="io b")
  parser.add_argument("-ic", "-ioc", type=float, required=False, default=0.0, help="io c")
  parser.add_argument("-vw", "-vcpuw", type=float, required=False, default=0.25, help="vcpu w")
  parser.add_argument("-cw", "-clkw", type=float, required=False, default=0.25, help="clk w")
  parser.add_argument("-mw", "-memw", type=float, required=False, default=0.25, help="mem w")
  parser.add_argument("-nw", "-netw", type=float, required=False, default=0.25, help="net w")
  parser.add_argument("-iw", "-iow", type=float, required=False, default=0.1, help="io w")
  parser.add_argument("-b", "-base", type=float, required=False, default=1.0, help="base performance")
  args = parser.parse_args()

  # initialyze analyzer
  analyzer = bayesian_optimizer_pool.BayesianOptimizerPool()
  request_str = "{\"appName\": \"mysql\", \"data\": [ ]}"
  request_dict = json.loads(request_str)
  session_id = "hyperpilot-sizing-demo-1-horray"

  analyzer.get_candidates(session_id, request_dict)
  print("...Initialized analyzer")
  bounds = util.get_feature_bounds(normalized=False)
  min_v = float(bounds[0][0])
  max_v = float(bounds[0][1])
  min_c = float(bounds[1][0])
  max_c = float(bounds[1][1])
  min_m = float(bounds[2][0])
  max_m = float(bounds[2][1])
  min_n = float(bounds[3][0])
  max_n = float(bounds[3][1])
  # min_i = float(bounds[4][0])
  # max_i = float(bounds[4][1])
  if min_v == 0.0:
    min_v = 0.2
  if min_m == 0.0:
    min_m = 0.5

  # initialize performance model
  global cloud
  global features
  global times
  global variance
  global aws_data
  global model
  cloud = CloudPerf(args.va, args.vb, args.vc, args.vw, max_v, \
                    args.ca, args.cb, args.cc, args.cw, max_c, \
                    args.ma, args.mb, args.mc, args.mw, max_m, \
                    args.na, args.nb, args.nc, args.nw, max_n, \
                    # args.ia, args.ib, args.ic, args.iw, max_i, \
                    args.b, args.noise, args.nrange)

  model = args.model
  if model == 'aws':
    print("Read app perf data for all aws instances from a json file")
    perf_file = "aws-all-instances-mysql.json"
    aws_data = AwsPerf(perf_file)

  print("Running analyzer harness with following parameters:")
  print(args)
  print()

  # build feature map for all the instances types
  all_nodetypes = list(util.get_all_nodetypes().values())
  numtypes = len(all_nodetypes)
  if numtypes < 3 * args.iter:
    print("ERROR: Not enough nodetypes in database")
    sys.exit()
  # build dictionary with features for all instances
  for nodetype in all_nodetypes:
    name = nodetype['name']
    np_feat = util.encode_nodetype(name)
    feat = np_feat.astype(type('float', (float,), {}))
    if feat[0] == 0.0 or feat[1] == 0.0 or feat[2] == 0.0 or feat[3] == 0.0:
      print("WARNING: problem with nodetype features ", name, feat)
    features[name] = feat
    times[name] = nodetype['cost']['time']
    variance[name] = nodetype['cost']['variance']
  # visited instances
  visited = set()
  print("...Got information for %d instance types" %numtypes)

  # main loop
  for i in range(args.iter): #range(args.iter):
    print("...Iteration %d out of %d" %(i, args.iter))
    # check if done
    while True:
      status_dict = analyzer.get_status(session_id).to_dict()
      if status_dict['status'] != "running":
        break
      time.sleep(1)

    if args.verbose:
      print("......Analyzer status returned:", status_dict)
    # throw error if needed
    if status_dict["status"] != "done":
      print("ERROR: Analyzer returned with status %s"  %status_dict["status"])
      sys.exit()
    # termination
    if len(status_dict["data"]) == 0:
      print("...Terminating due to empty reply")
      break

    # prepare next candidates
    lastApp = status_dict["data"][len(status_dict["data"])-1]
    request_str = "{\"appName\": %d, \"data\": [ " %lastApp
    count = 0
    # print('status-dict:', status_dict["data"])
    for nodetype in status_dict["data"]:
      count += 1
      feat = features[nodetype]
      if model == 'aws': # use real data from aws
          perf = aws_data.perf_map[nodetype]['qosValue']
      else: # use CloudPerf model instead
          perf = times[nodetype] #+ np.sqrt(variance[nodetype]) * np.random.randn()
          # perf = cloud.perf(feat[0], feat[1], feat[2], feat[3], args.noise)
      request_str += "{\"instanceType\": %d, \"qosValue\": %f, \"variance\": %f}" %(nodetype, perf, variance[nodetype])
      if count < len(status_dict["data"]):
        request_str += ", "
      if nodetype in visited:
        print("WARNING: re-considering type %s" %nodetype)
      else:
        visited.add(nodetype)
        #print("......Considering nodetype %s" %nodetype)
    request_str += "]}"
    request_dict = json.loads(request_str)
    result_dict = analyzer.get_candidates(session_id, request_dict).to_dict()
    if args.verbose:
      print("......Called analyzer with arguments:", request_str)
      print("........Result returned:", result_dict)
    if result_dict['status'] == "done":
      print("......Sizing session is complete")
      break
    time.sleep(1)

  # evaluate results
  slo = float(util.get_slo_value("mysql"))
  budget = float(util.get_budget("mysql"))

  optPC = {}
  optP = {}
  optC = {}
  #alltypes = {}
  for nodetype in features:
    feat = features[nodetype]
    if model == 'aws': # use real data from aws
      perf = aws_data.perf_map[nodetype]['qosValue']
    else: # use CloudPerf model instead
      # perf = cloud.perf(feat[0], feat[1], feat[2], feat[3], noise)
      perf = times[nodetype]

    price = util.get_price(nodetype)
    cost = util.compute_cost(price, 'throughput', perf)
    optPC[nodetype] = perf/cost
    if cost <= budget:
      optP[nodetype] = perf
    if perf >= slo:
      optC[nodetype] = cost
    #alltypes[nodetype] = [cost, perf, perf/cost]

  sorted_optPC = sorted(optPC.items(), key=operator.itemgetter(1), reverse=True)
  sorted_optP = sorted(optP.items(), key=operator.itemgetter(1), reverse=True)
  sorted_optC = sorted(optC.items(), key=operator.itemgetter(1), reverse=False)

  #alltypes_sorted = sorted(alltypes.items(), key=operator.itemgetter(1), reverse=False)
  #with open('aws-all-instances-sorted.json', 'w') as fp:
  #  json.dump(alltypes_sorted, fp)

  # print results
  print("")
  print("")
  print(".......................")
  print("... Analyzer results...")
  print("Iterations requested): %d" %args.iter)
  print("Noise: ", args.noise)
  print("Noise range: ", args.nrange)
  print("Min perf: ", slo)
  print("Max cost: ", budget)
  print("Nodetypes examined: ", len(visited))
  print("")
  print("Performance/cost")
  i = 0
  for key, _ in sorted_optPC:
    if i < 5:
      print("   Available #%2d: %11s, %s" %(i+1, key, nodeinfo(key)))
    if key in visited:
      print("   Found    #%2d:  %11s, %s" %(i+1, key, nodeinfo(key)))
      break
    i += 1
  print("")
  '''
  print("Cost with performance constraint (%0.2f)" %slo)
  i = 0
  for key, _ in sorted_optC:
    if i < 5:
      print("   Available #%2d: %11s, %s" %(i+1, key, nodeinfo(key)))
    if key in visited:
      print("   Found    #%2d:  %11s, %s" %(i+1, key, nodeinfo(key)))
      break
    i += 1
  print("")
  print("Performance with cost constraint (%0.2f)" %budget)
  i = 0
  for key, _ in sorted_optP:
    if i < 5:
      print("   Available #%2d: %11s, %s" %(i+1, key, nodeinfo(key)))
    if key in visited:
      print("   Found    #%2d:  %11s, %s" %(i+1, key, nodeinfo(key)))
      break
    i += 1
  '''

__main__()
