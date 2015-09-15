package dk.gp.cogp.lb

import breeze.linalg.DenseMatrix
import scala.collection._
import breeze.linalg.cholesky
import dk.gp.math.invchol
import dk.gp.cogp.model.CogpModel
import breeze.linalg.DenseVector

class LowerBound(val model: CogpModel, val x: DenseMatrix[Double], val y: DenseMatrix[Double]) {

  private val kZZjMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()
  private val kZZjInvMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()
  private val kXZjMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()

  private val kZZiMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()
  private val kZZiInvMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()
  private val kXZiMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()

  def kZZj(j: Int): DenseMatrix[Double] = kZZjMap.getOrElseUpdate(j, calckZZj(j))
  def kZZjInv(j: Int): DenseMatrix[Double] = kZZjInvMap.getOrElseUpdate(j, calckZZjInv(j))
  def kXZj(j: Int): DenseMatrix[Double] = kXZjMap.getOrElseUpdate(j, calckXZj(j))

  def kZZi(i: Int): DenseMatrix[Double] = kZZiMap.getOrElseUpdate(i, calckZZi(i))
  def kZZiInv(i: Int): DenseMatrix[Double] = kZZiInvMap.getOrElseUpdate(i, calckZZiInv(i))
  def kXZi(i: Int): DenseMatrix[Double] = kXZiMap.getOrElseUpdate(i, calckXZi(i))

  def calcAi(i: Int): DenseMatrix[Double] = kXZi(i) * kZZiInv(i)
  def calcAj(j: Int): DenseMatrix[Double] = kXZj(j) * kZZjInv(j)

  def yi(i:Int):DenseVector[Double] = {
    y(::, i)
  }

  private def calckZZj(j: Int): DenseMatrix[Double] = {

    val z = model.g(j).z
    val covFunc = model.g(j).covFunc
    val covFuncParams = model.g(j).covFuncParams

    covFunc.cov(z, z, covFuncParams) + 1e-10 * DenseMatrix.eye[Double](z.size)
  }

  private def calckZZjInv(j: Int): DenseMatrix[Double] = invchol(cholesky(kZZj(j)).t)

  private def calckXZj(j: Int): DenseMatrix[Double] = {

    val z = model.g(j).z

    model.g(j).covFunc.cov(x, z, model.g(j).covFuncParams)
  }

  private def calckZZi(i: Int): DenseMatrix[Double] = {

    val z = model.h(i).z
    val covFunc = model.h(i).covFunc
    val covFuncParams = model.h(i).covFuncParams

    covFunc.cov(z, z, covFuncParams) + 1e-10 * DenseMatrix.eye[Double](z.size)
  }

  private def calckZZiInv(i: Int): DenseMatrix[Double] = invchol(cholesky(kZZi(i)).t)

  private def calckXZi(i: Int): DenseMatrix[Double] = {

    val z = model.h(i).z

    model.h(i).covFunc.cov(x, z, model.h(i).covFuncParams)
  }
}

object LowerBound {
  def apply(model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): LowerBound = new LowerBound(model, x, y)
}