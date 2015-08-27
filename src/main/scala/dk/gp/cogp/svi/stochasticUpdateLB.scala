package dk.gp.cogp.svi

import breeze.linalg.DenseMatrix
import dk.gp.cogp.svi.v.stochasticUpdateV
import dk.gp.cogp.svi.u.stochasticUpdateU
import dk.gp.cov.CovFunc
import dk.gp.cogp.svi.w.stochasticUpdateW
import dk.gp.cogp.svi.beta.stochasticUpdateBeta
import breeze.linalg.DenseVector
import dk.gp.cov.utils.covDiag
import dk.gp.cogp.CogpModel
import dk.gp.cogp.svi.hypcovg.stochasticUpdateHypCovG
import dk.gp.cogp.svi.hypcovh.stochasticUpdateHypCovH

object stochasticUpdateLB {

  def apply(model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double], l: Double): CogpModel = {

    val newU = (0 until model.g.size).map { j => stochasticUpdateU(j, model, x, y) }.toArray
    val newHypCovG: Array[(DenseVector[Double], DenseVector[Double])] = (0 until model.g.size).map { j => stochasticUpdateHypCovG(j, model, x, y) }.toArray
    val newG = (0 until model.g.size).map { j =>
      model.g(j).copy(u = newU(j), covFuncParams = newHypCovG(j)._1, covFuncParamsDelta = newHypCovG(j)._2)
    }.toArray

    val newV = (0 until model.h.size).map { i => stochasticUpdateV(i, model, x, y) }.toArray
    val newHypCovH: Array[(DenseVector[Double], DenseVector[Double])] = (0 until model.h.size).map { i => stochasticUpdateHypCovH(i, model, x, y) }.toArray
    val newH = (0 until model.h.size).map { i =>
      model.h(i).copy(u = newV(i), covFuncParams = newHypCovH(i)._1, covFuncParamsDelta = newHypCovH(i)._2)
    }.toArray

    val (newW, newWDelta) = stochasticUpdateW(model, x, y)
    val (newBeta, newBetaDelta) = stochasticUpdateBeta(model, x, y)

    val newModel = model.copy(g = newG, h = newH, beta = newBeta, betaDelta = newBetaDelta, w = newW, wDelta = newWDelta)

    newModel
  }

  // private def
}