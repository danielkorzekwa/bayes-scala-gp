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

object createSingleHZeroGModel {
  
  def apply(tasks: Array[Task], z: DenseVector[Double]): CogpModel = apply(tasks,z.toDenseMatrix.t)
  def apply(tasks: Array[Task], z: DenseMatrix[Double]): CogpModel = {

    val hVariable0 = CogpGPVar(y = tasks(0).y, z,covFunc = CovSEiso(), covFuncParams = DenseVector(log(1), log(1)))

    //likelihood noise precision
    val beta = DenseVector(1d / 0.01) // [P x 1]

    //mixing weights
    val w = new DenseMatrix[Double](0,0)

    val model = CogpModel( Array(), Array(hVariable0), beta, w)
    model
  }

 

}