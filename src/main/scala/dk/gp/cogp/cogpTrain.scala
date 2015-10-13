package dk.gp.cogp

import breeze.linalg.DenseMatrix
import dk.gp.cogp.model.CogpModel
import dk.gp.cogp.svi.stochasticUpdateCogpModel
import breeze.linalg.DenseVector
import dk.gp.cogp.lb.LowerBound
import com.typesafe.scalalogging.slf4j.LazyLogging
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.model.Task
import breeze.numerics._
import breeze.linalg._
import breeze.stats._

object cogpTrain extends LazyLogging {

  private val defaultProgressListener = (currLb: LowerBound, currIter: Int, maxIterNum: Int) => {
    logger.info("iter=%d/%d, llh=%.3f".format(currIter, maxIterNum, calcLBLoglik(currLb)))
  }

  private val defaultConfig = CogpConfig(trainCovParams = true)

  /**
   * @param tasks
   * @param model
   * @param cogpConfig
   * @param progressListener (current lower bound, currIter,maxIter):Unit
   */
  def apply(tasks: Array[Task], model: CogpModel, iterNum: Int, cogpConfig: CogpConfig = defaultConfig, progressListener: (LowerBound, Int, Int) => Unit = defaultProgressListener): CogpModel = {

    val lowerBound = LowerBound(model, tasks)

    val finalLB = (0 until iterNum).foldLeft(lowerBound) {
      case (currLB, iter) =>
        val newLB = stochasticUpdateCogpModel(currLB, tasks,cogpConfig.trainCovParams)

        progressListener(newLB, iter, iterNum)

        newLB
    }
    finalLB.model
  }

  private def format(x: DenseVector[Double]): String = x.toArray.map(x => "%.3f".format(x)).toList.toString
}