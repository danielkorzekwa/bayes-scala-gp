package dk.gp.cogp.lb.grad

import breeze.linalg.DenseMatrix
import breeze.linalg.cholesky
import dk.gp.math.invchol
import dk.gp.cogp.lb.LowerBound
import breeze.linalg.InjectNumericOps
import dk.gp.cogp.model.CogpModel

object calcLBGradVEta2 {

  def apply(i: Int, lowerBound:LowerBound,model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {
    
    val kZZiInv = lowerBound.kZZiInv(i)

    val Ai = lowerBound.calcAi(i)

    val v = model.h.map(_.u)
    val beta = model.beta

    val lambda = kZZiInv + beta(i) * Ai.t * Ai

    val vCholR = cholesky(v(i).v).t
    val vInv = invchol(vCholR)

    val grad = 0.5 * vInv - 0.5 * lambda

    grad
  }

}