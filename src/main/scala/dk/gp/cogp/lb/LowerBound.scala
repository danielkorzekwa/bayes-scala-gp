package dk.gp.cogp.lb

import breeze.linalg.DenseMatrix
import scala.collection._
import breeze.linalg.cholesky
import dk.gp.math.invchol
import dk.gp.cogp.model.CogpModel
import breeze.linalg.DenseVector
import dk.gp.cov.utils.covDiagD
import dk.gp.cov.utils.covDiag
import breeze.linalg._
import dk.gp.cogp.model.Task

class LowerBound(val model: CogpModel, val tasks: Array[Task]) {

  private val kZZjMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()
  private val kZZjInvMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()
  private val kXZjMap: mutable.Map[Int, mutable.Map[Int, DenseMatrix[Double]]] = mutable.Map()

  private val kZZiMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()
  private val kZZiInvMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()
  private val kXZiMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()

  private val dkZZjMap: mutable.Map[Int, Array[DenseMatrix[Double]]] = mutable.Map()
  private val dkXZjMap: mutable.Map[Int, mutable.Map[Int, Array[DenseMatrix[Double]]]] = mutable.Map()

  private val dkZZiMap: mutable.Map[Int, Array[DenseMatrix[Double]]] = mutable.Map()
  private val dkXZiMap: mutable.Map[Int, Array[DenseMatrix[Double]]] = mutable.Map()

  def kZZj(j: Int): DenseMatrix[Double] = kZZjMap.getOrElseUpdate(j, model.g(j).calckZZ())
  def kZZjInv(j: Int): DenseMatrix[Double] = kZZjInvMap.getOrElseUpdate(j, calckZZjInv(j))
  def kXZj(i: Int, j: Int): DenseMatrix[Double] = {
    val x = tasks(i).x
    val kXZ = kXZjMap.getOrElseUpdate(j, mutable.Map()).getOrElseUpdate(i, model.g(j).calckXZ(x))
    kXZ
  }

  def kZZi(i: Int): DenseMatrix[Double] = kZZiMap.getOrElseUpdate(i, model.h(i).calckZZ())
  def kZZiInv(i: Int): DenseMatrix[Double] = kZZiInvMap.getOrElseUpdate(i, calckZZiInv(i))
  def kXZi(i: Int): DenseMatrix[Double] = {
    val x = tasks(i).x
    val kXZ = kXZiMap.getOrElseUpdate(i, model.h(i).calckXZ(x))
    kXZ
  }

  def dKzzj(j: Int, k: Int): DenseMatrix[Double] = dkZZjMap.getOrElseUpdate(j, model.g(j).calcdKzz())(k)
  def dKxzj(i: Int, j: Int, k: Int): DenseMatrix[Double] = {
    val x = tasks(i).x
    val dKxz = dkXZjMap.getOrElseUpdate(j, mutable.Map()).getOrElseUpdate(i, model.g(j).calcdKxz(x))(k)
    dKxz
  }

  def dKzzi(i: Int, k: Int): DenseMatrix[Double] = dkZZiMap.getOrElseUpdate(i, model.h(i).calcdKzz())(k)
  def dKxzi(i: Int, k: Int): DenseMatrix[Double] = {
    val x = tasks(i).x
    val dKxz = dkXZiMap.getOrElseUpdate(i, model.h(i).calcdKxz(x))(k)
    dKxz
  }

  def calcKxxDiagi(i: Int): DenseVector[Double] = {

    val x = tasks(i).x
    val kXXiDiag = covDiag(x, model.h(i).covFunc, model.h(i).covFuncParams)
    kXXiDiag
  }

  def calcKxxDiagj(i: Int, j: Int): DenseVector[Double] = {
    val x = tasks(i).x
    val kXXDiag = covDiag(x, model.g(j).covFunc, model.g(j).covFuncParams)
    kXXDiag
  }

  def calcAi(i: Int): DenseMatrix[Double] = kXZi(i) * kZZiInv(i)
  def calcAj(i: Int, j: Int): DenseMatrix[Double] = kXZj(i, j) * kZZjInv(j)

  /**
   * @param k k-derivative
   */
  def calcdKxxDiagj(i: Int, j: Int, k: Int): DenseVector[Double] = {
    val x = tasks(i).x
    val dKxxDiagArray = covDiagD(x, model.g(j).covFunc, model.g(j).covFuncParams)
    dKxxDiagArray(k)
  }

  def calcdKxxDiagi(i: Int, k: Int): DenseVector[Double] = {
    val x = tasks(i).x
    val dKxxDiagArray = covDiagD(x, model.h(i).covFunc, model.h(i).covFuncParams)
    dKxxDiagArray(k)
  }

  def calcdAj(i: Int, j: Int, k: Int): DenseMatrix[Double] = {
    val kZZinv = kZZjInv(j)
    val AjD = dKxzj(i, j, k) * kZZinv - calcAj(i, j) * dKzzj(j, k) * kZZinv
    AjD
  }

  def calcdAi(i: Int, k: Int): DenseMatrix[Double] = {
    val kZZinv = kZZiInv(i)
    val dAi = dKxzi(i, k) * kZZinv - calcAi(i) * dKzzi(i, k) * kZZinv
    dAi
  }

  def yi(i: Int): DenseVector[Double] = tasks(i).y

  private def calckZZjInv(j: Int): DenseMatrix[Double] = invchol(cholesky(kZZj(j)).t)
  private def calckZZiInv(i: Int): DenseMatrix[Double] = invchol(cholesky(kZZi(i)).t)

}

object LowerBound {
  def apply(model: CogpModel, tasks: Array[Task]): LowerBound = new LowerBound(model, tasks)

}