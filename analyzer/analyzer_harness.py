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
import sys
import time
from . import util
from . import bayesian_optimizer_pool

# a couple of globals
features = {}
cloud = None

class CloudPerf(object):
  """ A class for a cloud performance model
  """

  def __init__(self,
               vcpu_a, vcpu_b, vcpu_c, vcpu_w,
               clk_a, clk_b, clk_c, clk_w,
               mem_a, mem_b, mem_c, mem_w,
               net_a, net_b, net_c, net_w,
               io_a, io_b, io_c, io_w,
               base, noise, nrange):
    """ initialize key parameters
    """
    self.vcpu_a = vcpu_a
    self.vcpu_b = vcpu_b
    self.vcpu_c = vcpu_c
    self.clk_a = clk_a
    self.clk_b = clk_b
    self.clk_c = clk_c
    self.mem_a = mem_a
    self.mem_b = mem_b
    self.mem_c = mem_c
    self.net_a = net_a
    self.net_b = net_b
    self.net_c = net_c
    self.io_a = io_a
    self.io_b = io_b
    self.io_c = io_c
    self.vcpu_w = vcpu_w
    self.clk_w = clk_w
    self.mem_w = mem_w
    self.net_w = net_w
    self.io_w = io_w
    self.noise = noise
    self.nrange = nrange
    self.base = base
    # some sanity checks
    if (vcpu_w + clk_w + mem_w + net_w + io_w) != 1.0:
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
    if io_a == 0 and io_b == 0 and io_c == 0 and io_w != 0:
      print("ERROR: IO performance parameters are all 0")
      sys.exit()


  def vcpu_model(self, vcpu):
    """ a vcpu based performance model.
        perf = vcpu / (a + vcpu*b + vcpu^2*b)
        interesting cases:  b=c=0, b>0/c=0, b>>0/c=0, b>>0/c>0
        assumes vcpu_min = 1
    """
    return vcpu / (self.vcpu_a + self.vcpu_b*vcpu + self.vcpu_c*vcpu*vcpu)

  def clk_model(self, clk):
    """ a clk based performance model.
        perf = clk / (a + clk*b + clk^2*b)
        assumes clk_min = 2.3
    """
    clk = (clk / 2.3)
    return clk / (self.clk_a + self.clk_b*clk + self.clk_c*clk*clk)

  def mem_model(self, mem):
    """ a mem based performance model.
        perf = mem / (a + mem*b + mem^2*b)
        interesting cases:  b=c=0, b>0/c=0, b>>0/c=0, b>>0/c>0
        assumes mem_min = 0.5
    """
    mem = (mem / 0.5)
    return mem / (self.mem_a + self.mem_b*mem + self.mem_c*mem*mem)

  def net_model(self, net):
    """ a net based performance model.
        perf = net / (a + net*b + net^2*b)
        interesting cases:  b=c=0, b>0/c=0, b>>0/c=0, b>>0/c>0
        assumes net_min = 100
    """
    net = (net / 100)
    return net / (self.net_a + self.net_b*net + self.net_c*net*net)

  def io_model(self, io):
    """ a io based performance model.
        perf = io / (a + io*b + io^2*b)
        interesting cases:  b=c=0, b>0/c=0, b>>0/c=0, b>>0/c>0
        assumes io_min = 50
    """
    io = (io / 50)
    return io / (self.io_a + self.io_b*io + self.io_c*io*io)

  def perf(self, vcpu, clk, mem, net, io, noise):
    """ estimate performance
    """
    perf = self.vcpu_w * self.vcpu_model(vcpu) + \
           self.clk_w * self.clk_model(clk) + \
           self.mem_w * self.mem_model(mem) + \
           self.net_w * self.net_model(net) + \
           self.io_w * self.io_model(io)
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
  if nodetype == "none":
    return "perf %9.2f, price %5.3f, cost %6.2f, perf/cost %8.2f" \
         %(0.0, 0.0, 0.0, 0.0)
  feat = features[nodetype]
  perf = cloud.perf(feat[0], feat[1], feat[2], feat[3], feat[4], False)
  price = util.get_price(nodetype)
  cost = util.compute_cost(price, 'throughput', perf)
  return "perf %9.2f, price %5.3f, cost %6.2f, perf/cost %8.2f" \
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
  parser.add_argument("-i", "--iter", type=int, required=False, default=10, help="maximum iterations")
  parser.add_argument("-n", "--noise", type=str2bool, required=False, default=True, help="add noise to cloud performance")
  parser.add_argument("-r", "--nrange", type=int, required=False, default=10, help="noise range (int)")
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
  parser.add_argument("-vw", "-vcpuw", type=float, required=False, default=1.0, help="vcpu w")
  parser.add_argument("-cw", "-clkw", type=float, required=False, default=0.0, help="clk w")
  parser.add_argument("-mw", "-memw", type=float, required=False, default=0.0, help="mem w")
  parser.add_argument("-nw", "-netw", type=float, required=False, default=0.0, help="net w")
  parser.add_argument("-iw", "-iow", type=float, required=False, default=0.0, help="io w")
  parser.add_argument("-b", "-base", type=float, required=False, default=1.0, help="base performance")
  args = parser.parse_args()

  # initialize performance model
  global cloud
  global features
  cloud = CloudPerf(args.va, args.vb, args.vc, args.vw, \
                         args.ca, args.cb, args.cc, args.cw, \
                         args.ma, args.mb, args.mc, args.mw, \
                         args.na, args.nb, args.nc, args.nw, \
                         args.ia, args.ib, args.ic, args.iw, \
                         args.b, args.noise, args.nrange)
  print("Running analyzer harness with following parameters:")
  print(args)
  print()

  # get all the instance info
  all_nodetypes = util.get_all_nodetypes()['data']
  numtypes = len(all_nodetypes)
  if numtypes < args.iter*3:
    print("ERROR: Not enough nodetypes in database")
    sys.exit()
  # build dictionary with features for all instances
  for nodetype in all_nodetypes:
    name = nodetype['name']
    np_feat = util.encode_instance_type(name)
    feat = np_feat.astype(type('float', (float,), {}))
    if feat[0] == 0.0 or feat[1] == 0.0 or feat[2] == 0.0 or feat[3] == 0.0 or feat[4] == 0.0:
      print("WARNING: problem with nodetype features ", name, feat)
    features[name] = feat
  # visited instances
  visited = set()
  print("...Got information for %d instance types" %numtypes)

  # initialyze analyzer
  analyzer = bayesian_optimizer_pool.BayesianOptimizerPool.instance()
  request_str = "{\"appName\": \"redis\", \"data\": [ ]}"
  request_dict = json.loads(request_str)
  analyzer.get_candidates("redis", request_dict)
  print("...Initialized analyzer")

  #main loop
  for i in range(args.iter):
    print("...Iteration %d out of %d" %(i, args.iter))
    # check if done
    while True:
      status_dict = analyzer.get_status("redis")
      if status_dict['status'] != "running":
        break
      time.sleep(1)
    if args.verbose:
      print("......Analyzer returned ", status_dict)
    # throw error if needed
    if status_dict["status"] != "done":
      print("ERROR: Analyzer returned with status %s"  %status_dict["status"])
      sys.exit()
    # termination
    if len(status_dict["data"]) == 0:
      print("...Terminating due to empty reply")
      break
    # prepare next candidates
    request_str = "{\"appName\": \"redis\", \"data\": [ "
    count = 0
    for nodetype in status_dict["data"]:
      count += 1
      feat = features[nodetype]
      perf = cloud.perf(feat[0], feat[1], feat[2], feat[3], feat[4], True)
      request_str += "{\"instanceType\": \"%s\", \"qosValue\": %f}" %(nodetype, perf)
      if count < len(status_dict["data"]):
        request_str += ", "
      if nodetype in visited:
        print("WARNING: re-considering type %s" %nodetype)
      else:
        visited.add(nodetype)
        print("......Considering nodetype %s" %nodetype)
    request_str += "]}"
    request_dict = json.loads(request_str)
    analyzer.get_candidates("redis", request_dict)
    if args.verbose:
      print("......Called analyzer with arguments %s", request_str)
    time.sleep(1)

  # evaluate results
  slo = float(util.get_slo_value("redis"))
  budget = float(util.get_budget("redis"))
  # scan all visited nodetypes
  v_optPC_i = v_optP_i = v_optC_i = "none"
  v_optP_perf = v_optC_perf = 0.0
  v_optP_cost = v_optC_cost = 1.79e+30
  v_optPC_perfcost = 0.0
  for key in visited:
    feat = features[key]
    perf = cloud.perf(feat[0], feat[1], feat[2], feat[3], feat[4], False)
    price = util.get_price(key)
    cost = util.compute_cost(price, 'throughput', perf)
    # best perf under budget constraint
    if cost <= budget and (perf > v_optP_perf or (perf == v_optP_perf and cost < v_optP_cost)):
      v_optP_i = key
      v_optP_perf = perf
      v_optP_cost = cost
    # best cost under perf constraint
    if perf >= slo and (cost < v_optC_cost or (cost == v_optC_cost and perf > v_optC_perf)):
      v_optC_i = key
      v_optC_cost = cost
      v_optC_perf = perf
    # best perf/cost
    if (perf/cost) > v_optPC_perfcost:
      v_optPC_i = key
      v_optPC_perfcost = perf/cost
  # scan all nodetypes
  c_optPC_i = c_optP_i = c_optC_i = "none"
  c_optP_perf = c_optC_perf = 0.0
  c_optP_cost = c_optC_cost = 1.79e+30
  c_optPC_perfcost = 0.0
  for key in features:
    feat = features[key]
    perf = cloud.perf(feat[0], feat[1], feat[2], feat[3], feat[4], False)
    price = util.get_price(key)
    cost = util.compute_cost(price, 'throughput', perf)
    # best perf under budget constraint
    if cost <= budget and (perf > c_optP_perf or (perf == c_optP_perf and cost < c_optP_cost)):
      c_optP_i = key
      c_optP_perf = perf
      c_optP_cost = cost
    # best cost under perf constraint
    if perf >= slo and (cost < c_optC_cost or (cost == c_optC_cost and perf > c_optC_perf)):
      c_optC_i = key
      c_optC_cost = cost
      c_optC_perf = perf
    # best perf/cost
    if (perf/cost) > c_optPC_perfcost:
      c_optPC_i = key
      c_optPC_perfcost = perf/cost

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
  print("   Best available: %10s, %s" %(c_optPC_i, nodeinfo(c_optPC_i)))
  print("   Best found:     %10s, %s" %(v_optPC_i, nodeinfo(v_optPC_i)))
  print("")
  print("Cost with performance constraint (%0.2f)" %slo)
  print("   Best available: %10s, %s" %(c_optC_i, nodeinfo(c_optC_i)))
  print("   Best found:     %10s, %s" %(v_optC_i, nodeinfo(v_optC_i)))
  print("")
  print("Performance with cost constraint (%0.2f)" %budget)
  print("   Best available: %10s, %s" %(c_optP_i, nodeinfo(c_optP_i)))
  print("   Best found:     %10s, %s" %(v_optP_i, nodeinfo(v_optP_i)))

__main__()
