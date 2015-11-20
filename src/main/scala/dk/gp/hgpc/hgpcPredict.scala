package dk.gp.hgpc

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.math.UnivariateGaussian
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.gp.math.MultivariateGaussian
import dk.gp.gp.gpPredictSingle
import dk.bayes.math.gaussian.Gaussian
import breeze.numerics._
import dk.bayes.infer.epnaivebayes.EPNaiveBayesFactorGraph
import dk.gp.hgpc.util.TaskVariable
import dk.bayes.dsl.factor._
import dk.bayes.math.gaussian.canonical.CanonicalGaussian

/**
 * Hierarchical Gaussian Process classification. Multiple Gaussian Processes for n tasks with a single shared parent GP.
 */
object hgpcPredict {

  case class TaskPosterior(x: DenseMatrix[Double], xPosterior: DenseCanonicalGaussian)

  /**
   * Returns vector of probabilities for a class 1
   */
  def apply(t: DenseMatrix[Double], model: HgpcModel): DenseVector[Double] = {
    val taskPosteriorByTaskId: Map[Int, TaskPosterior] = createTaskPosteriorByTaskId(t, model)

    val predictedArray = (0 until t.rows).par.map { rowIndex =>

      val tRow = t(rowIndex, ::).t
      val taskId = tRow(0).toInt
      val taskPosterior = taskPosteriorByTaskId(taskId)

      val tTestPrior = gpPredictSingle(tRow.toDenseMatrix, MultivariateGaussian(taskPosterior.xPosterior.mean, taskPosterior.xPosterior.variance), taskPosterior.x, model.covFunc, model.covFuncParams, model.mean)
      val predictedProb = Gaussian.stdCdf(tTestPrior.m(0) / sqrt(1d + tTestPrior.v(0, 0)))

      predictedProb
    }.toArray
    DenseVector(predictedArray)
  }

  private def createTaskPosteriorByTaskId(xTest: DenseMatrix[Double], model: HgpcModel): Map[Int, TaskPosterior] = {
    
     val covU = model.covFunc.cov(model.u, model.u, model.covFuncParams) + DenseMatrix.eye[Double](model.u.rows) * 1e-7
    val meanU = DenseVector.zeros[Double](model.u.rows) + model.mean

    val uVariable = dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian(meanU, covU)

      val taskIds = model.x(::, 0).toArray.distinct
      val taskVariables = taskIds.map{ taskId => 
    
          val idx = model.x(::, 0).findAll { x => x == taskId }
      val taskX = model.x(idx, ::).toDenseMatrix
      val taskY = model.y(idx).toDenseVector
        
        TaskVariable(taskX,taskY,model,uVariable).asInstanceOf[DoubleFactor[CanonicalGaussian, _]] //@TODO this is a hack to allow for using a proper implicit for multOp()
     }
      
      
    val factorGraph = EPNaiveBayesFactorGraph(uVariable, taskVariables, true)
    
    factorGraph.calibrate(maxIter = 10, threshold = 1e-4)
    
    throw new UnsupportedOperationException("Not implemented yet")
  }
}