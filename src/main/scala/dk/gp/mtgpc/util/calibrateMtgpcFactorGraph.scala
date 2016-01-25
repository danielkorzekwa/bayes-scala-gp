package dk.gp.mtgpc.util

import dk.gp.gpc.util.calibrateGpcFactorGraph

object calibrateMtgpcFactorGraph {

  def apply(factorGraph: MtgpcFactorGraph, maxIter: Int) = {
    factorGraph.taskFactorGraphs.par.foreach(factorGraph => calibrateGpcFactorGraph(factorGraph, maxIter))
  }
}