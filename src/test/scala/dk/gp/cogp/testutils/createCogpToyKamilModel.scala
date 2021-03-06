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

object createCogpToyKamilModel {

  def apply(tasks: Array[Task], z: DenseMatrix[Double]): CogpModel = {

    val allY = DenseVector.vertcat(tasks.map(t => t.y): _*)

    val gVariable = CogpGPVar(y = allY, z, covFunc = CovSEiso(), covFuncParams = DenseVector(log(1), log(1)))

    val hVariable0 = CogpGPVar(y = tasks(0).y, z, covFunc = CovSEiso(), covFuncParams = DenseVector(log(1), log(1)))
    val hVariable1 = CogpGPVar(y = tasks(1).y, z, covFunc = CovSEiso(), covFuncParams = DenseVector(log(1), log(1)))

    //likelihood noise precision
    val beta = DenseVector(1d / 0.01, 1d / 0.01) // [P x 1]

    //mixing weights
    val w = new DenseMatrix(2, 1, Array(1.0, 1)) // [P x Q]

    val model = CogpModel(Array(gVariable), Array(hVariable0, hVariable1), beta, w)
    model
  }

}