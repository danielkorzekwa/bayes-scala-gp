package dk.gp.cogp.lb

import breeze.linalg.DenseVector
import breeze.linalg._
import breeze.numerics._

object wAm {

  def apply(i: Int, lb: LowerBound): DenseVector[Double] = {

    val wAm = lb.model.g.zipWithIndex.map {
      case (g, j) =>

        val Aj = lb.calcAj(i, j)
        lb.model.w(i, j) * Aj * g.u.m
    }.toSeq

    if (wAm.size > 0) sum(wAm)
    else DenseVector.zeros(lb.yIdx(i).size)
  }
}