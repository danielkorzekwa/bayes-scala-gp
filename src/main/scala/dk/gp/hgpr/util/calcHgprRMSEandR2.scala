package dk.gp.hgpr.util

import dk.gp.hgpr.HgprModel
import breeze.linalg.cholesky
import dk.gp.gp.GPPredictSingle
import dk.gp.gpr.gprLoglik
import dk.bayes.math.gaussian.MultivariateGaussian
import dk.gp.math.invchol
import breeze.linalg.DenseMatrix
import breeze.numerics._
import breeze.linalg.DenseVector
import dk.bayes.math.accuracy.rmse
import breeze.stats._
import breeze.linalg._

object calcHgprRMSEandR2 {

  /**
   * 
   * @return [n, rmse,r2]
   */
  def apply(model: HgprModel,filter:(DenseVector[Double]) => Boolean = (x) => true): Tuple3[Int,Double, Double] = {

    val hgpFactorGraph = HgprFactorGraph(model.x, model.y, model.u, model.covFunc, model.covFuncParams, model.likNoiseLogStdDev)
    val uPosterior = hgpFactorGraph.calcUPosterior()

    val taskIds = model.x(::, 0).toArray.distinct

    val actualVsPredicted = taskIds.map { taskId =>

      val idx = model.x(::, 0).findAll { x => x == taskId }
      val taskX = model.x(idx, ::).toDenseMatrix
      val taskY = model.y(idx).toDenseVector

      val xFactorMsgUp = hgpFactorGraph.getXFactorMsgUp(taskId.toInt)

      val uVarMsgDown = uPosterior / xFactorMsgUp
      val xPrior = GPPredictSingle(MultivariateGaussian(uVarMsgDown.mean, uVarMsgDown.variance), model.u, model.covFunc, model.covFuncParams).predictSingle(taskX)
      val cPriorVarWithNoise = xPrior.v + DenseMatrix.eye[Double](taskX.rows) * exp(2 * model.likNoiseLogStdDev)

      val kXXInv = invchol(cholesky(cPriorVarWithNoise).t)

      val kXXInvY = kXXInv * taskY

      //e.q 5.11 from page 116, Carl Edward Rasmussen and Christopher K. I. Williams, The MIT Press, 2006
      val predictions = taskY.toArray.zipWithIndex.map {
        case (y_i, i) =>

          //  val yV = 1d / kXXInv(i, i)
          val yM = y_i - (kXXInvY)(i) / kXXInv(i, i)
          yM
      }
      DenseMatrix.horzcat(taskX,taskY.toDenseMatrix.t, DenseVector(predictions).toDenseMatrix.t)
    }

    val actualVsPredictedMat = DenseMatrix.vertcat(actualVsPredicted: _*)

    
    val filteredActualVsPredictedRows = (0 until actualVsPredictedMat.rows).map(i =>  actualVsPredictedMat(i,::).t).filter(r => filter(r(0 until r.size-2)))
    val filteredActualVsPredictedMat = DenseVector.horzcat(filteredActualVsPredictedRows :_*).t
    
    val actual = filteredActualVsPredictedMat(::,filteredActualVsPredictedMat.cols-2)
     val predicted = filteredActualVsPredictedMat(::,filteredActualVsPredictedMat.cols-1)
    val modelRMSE = rmse(actual, predicted)

    val predictedBaseline = DenseVector.fill(predicted.size)(mean(actual))
    val baselineRMSE = rmse(predictedBaseline, actual)
    val r2 = 1 - pow(modelRMSE, 2) / pow(baselineRMSE, 2)

    (predicted.size,modelRMSE, r2)

  }
}