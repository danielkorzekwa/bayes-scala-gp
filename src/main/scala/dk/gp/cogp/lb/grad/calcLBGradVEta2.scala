package dk.gp.cogp.lb.grad

import breeze.linalg.DenseMatrix
import breeze.linalg.cholesky
import dk.gp.math.invchol
import dk.gp.cogp.lb.LowerBound
import breeze.linalg.InjectNumericOps
import dk.gp.cogp.model.CogpModel
import breeze.linalg._

object calcLBGradVEta2 {

  def apply(i: Int, lb: LowerBound): DenseMatrix[Double] = {

    val kZZiInv = lb.kZZiInv(i)
    val Ai = lb.calcAi(i)

    val lambda = kZZiInv + lb.model.beta(i) * Ai.t * Ai

    val vInv = invchol(cholesky(lb.model.h(i).u.v).t)

    val grad = 0.5 * vInv - 0.5 * lambda

    grad
  }

}