package dk.gp.gpc.util

import dk.bayes.factorgraph2.api.calibrate
import dk.bayes.math.gaussian.canonical.CanonicalGaussian

object calibrateGpcFactorGraph {

  def apply(factorGraph: GpcFactorGraph, maxIter: Int): (Boolean, Int) = {

    /**
     * Update variables
     */
    factorGraph.fVariable.update()

    /**
     * Calibration
     */
    var calibrated = false
    def calibrateStep() = {
      val oldFMarginal = factorGraph.fVariable.get()

      factorGraph.yFactors.foreach(yFactor => yFactor.updateMsgV1())
      factorGraph.fVariable.update()

      calibrated = CanonicalGaussian.isIdentical(oldFMarginal, factorGraph.fVariable.get, 1e-4)
    }

    val (calib, iter) = calibrate(calibrateStep, maxIter, calibrated)

    (calib, iter)
  }
}