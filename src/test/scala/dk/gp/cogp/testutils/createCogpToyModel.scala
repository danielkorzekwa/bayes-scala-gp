package dk.gp.cogp.testutils

import breeze.linalg.DenseMatrix
import dk.gp.cov.CovSEiso
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import breeze.numerics._
import dk.gp.cov.CovNoise
import dk.gp.cogp.model.CogpGPVar
import breeze.stats._
import breeze.linalg._
import dk.gp.cogp.model.CogpModel
import dk.gp.cogp.model.Task
import breeze.stats._

object createCogpToyModel {

  def apply(tasks: Array[Task]): CogpModel = apply(tasks, tasks(0).x)

  def apply(tasks: Array[Task], z: DenseVector[Double]): CogpModel = apply(tasks, z.toDenseMatrix.t)

  def apply(tasks: Array[Task], z: DenseMatrix[Double]): CogpModel = {

    val yy = DenseVector.horzcat(tasks(0).y, tasks(1).y)

    val gVariable = CogpGPVar(y = mean(yy(*, ::)), z, covFunc = CovSEiso(), covFuncParams = DenseVector(log(1), log(1)))

    val hVariable0 = CogpGPVar(y = tasks(0).y, z, covFunc = CovNoise(), covFuncParams = DenseVector(log(1)))
    val hVariable1 = CogpGPVar(y = tasks(1).y, z, covFunc = CovNoise(), covFuncParams = DenseVector(log(1)))

    //likelihood noise precision
    val beta = DenseVector(1d / 0.01, 1d / 0.01) // [P x 1]

    //mixing weights
    val w = new DenseMatrix(2, 1, Array(1.0, 1)) // [P x Q]

    val model = CogpModel(Array(gVariable), Array(hVariable0, hVariable1), beta, w)
    model
  }

}