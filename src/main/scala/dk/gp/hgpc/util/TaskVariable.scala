package dk.gp.hgpc.util

import dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian
import dk.bayes.dsl.Variable
import breeze.linalg.DenseMatrix
import dk.gp.hgpc.HgpcModel
import breeze.linalg.DenseVector

case class TaskVariable(taskX:DenseMatrix[Double],taskY:DenseVector[Double],model:HgpcModel,val uVariable: MultivariateGaussian) extends Variable with TaskFactor {

  def getParents(): Seq[Variable] = List(uVariable)

}