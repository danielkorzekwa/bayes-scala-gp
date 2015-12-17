package dk.gp.hgpc.util

import dk.bayes.factorgraph2.api.calibrate
import dk.bayes.math.gaussian.canonical.CanonicalGaussian
import scala.util.Random
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian

object calibrateHgpcFactorGraph2 {

  def apply(factorGraph: HgpcFactorGraph2, maxIter: Int): (Boolean, Int) = {

    /**
     * Update variables
     */
    factorGraph.uVariable.update()
    factorGraph.taskVariablesMap.values.foreach(_.update())

    /**
     * Calibration
     */
    var calibrated = false
    def calibrateStep() = {

      val beforeUMarginal = factorGraph.uVariable.get()

      factorGraph.taskIds.foreach { taskId =>
        factorGraph.taskFactorsMap(taskId).updateMsgV2()

        factorGraph.taskVariablesMap(taskId).update()

        for (i <- 1 to 2) {
          factorGraph.taskYFactorsMap(taskId).foreach { taskYFactor => taskYFactor.updateMsgV1() }
          factorGraph.taskVariablesMap(taskId).update()
        }

        factorGraph.taskFactorsMap(taskId).updateMsgV1()
      }

      factorGraph.uVariable.update()

      calibrated = CanonicalGaussian.isIdentical(beforeUMarginal, factorGraph.uVariable.get, 1e-3)
    }

    val (calib, iter) = calibrate(calibrateStep, maxIter, calibrated)

    (calib, iter)
  }
}