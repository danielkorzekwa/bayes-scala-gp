package dk.gp.gpc.factorgraph2

import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.bayes.math.gaussian.canonical.CanonicalGaussian
import dk.bayes.math.gaussian.canonical.SparseCanonicalGaussian
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.bayes.math.gaussian.Gaussian
import dk.bayes.math.gaussian.canonical.SparseCanonicalGaussian
import breeze.linalg.CSCMatrix
import breeze.linalg.SparseVector

case class MvnGaussianThresholdFactor(v1: GaussianVariable, v2: BernVariable, xSize: Int, xIndex: Int, v: Double) extends DoubleFactor[CanonicalGaussian, Double] {

  def getV1(): Variable[CanonicalGaussian] = v1
  def getV2(): Variable[Double] = v2

  def getInitialMsgV1(): CanonicalGaussian = createZeroFactorMsgUp(xSize, Double.NaN)

  def calcNewMsgV1(): CanonicalGaussian = {

    val x = this.getV1.get.asInstanceOf[DenseCanonicalGaussian]
    val oldFactorMsgUp = this.getMsgV1.get.asInstanceOf[SparseCanonicalGaussian]
    val exceeds = this.getV2().get() == 1
    val oldYVarMsgUp = new DenseCanonicalGaussian(DenseMatrix(oldFactorMsgUp.k(xIndex, xIndex)), DenseVector(oldFactorMsgUp.h(xIndex)), oldFactorMsgUp.g)
    val factorMsgDown = (x.marginal(xIndex) / (oldYVarMsgUp)).toGaussian + Gaussian(0, v)

    //compute new factor msg up
    val projValue = (factorMsgDown).truncate(0, exceeds)

    val yVarMsgUp = (projValue / factorMsgDown) + Gaussian(0, v)

    val yVarMsgUpCanon = DenseCanonicalGaussian(yVarMsgUp.m, yVarMsgUp.v)

    val newFactorMsgUp = createZeroFactorMsgUp(xSize, yVarMsgUpCanon.g)
    newFactorMsgUp.k(xIndex, xIndex) = yVarMsgUpCanon.k(0, 0)
    newFactorMsgUp.h(xIndex) = yVarMsgUpCanon.h(0)
    newFactorMsgUp

  }

  def getInitialMsgV2(): Double = {
    Double.NaN
  }

  def calcNewMsgV2(): Double = {
    ???
  }

  private def createZeroFactorMsgUp(n: Int, g: Double): SparseCanonicalGaussian = {
    val newFactorMsgUpK = CSCMatrix.zeros[Double](n, n)
    val newFactorMsgUpH = SparseVector.zeros[Double](n)
    val newFactorMsgUp = new SparseCanonicalGaussian(newFactorMsgUpK, newFactorMsgUpH, g)

    newFactorMsgUp
  }
}

