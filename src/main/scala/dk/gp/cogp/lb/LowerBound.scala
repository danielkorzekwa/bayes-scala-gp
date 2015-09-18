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

class LowerBound(val model: CogpModel, val x: DenseMatrix[Double], val y: DenseMatrix[Double]) {

  val yIdx = y(::, *).map(yi => yi.findAll { !_.isNaN() }).t

  private val kZZjMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()
  private val kZZjInvMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()
  private val kXZjMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()

  private val kZZiMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()
  private val kZZiInvMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()
  private val kXZiMap: mutable.Map[Int, DenseMatrix[Double]] = mutable.Map()

  def kZZj(j: Int): DenseMatrix[Double] = kZZjMap.getOrElseUpdate(j, calckZZj(j))
  def kZZjInv(j: Int): DenseMatrix[Double] = kZZjInvMap.getOrElseUpdate(j, calckZZjInv(j))
  def kXZj(i: Int, j: Int): DenseMatrix[Double] = kXZjMap.getOrElseUpdate(j, calckXZj(j))(yIdx(i), ::).toDenseMatrix

  def kZZi(i: Int): DenseMatrix[Double] = kZZiMap.getOrElseUpdate(i, calckZZi(i))
  def kZZiInv(i: Int): DenseMatrix[Double] = kZZiInvMap.getOrElseUpdate(i, calckZZiInv(i))
  def kXZi(i: Int): DenseMatrix[Double] = kXZiMap.getOrElseUpdate(i, calckXZi(i))(yIdx(i), ::).toDenseMatrix

  def calcKxxDiagi(i: Int): DenseVector[Double] = {

    val kXXiDiag = covDiag(x, model.h(i).covFunc, model.h(i).covFuncParams)
    kXXiDiag(yIdx(i)).toDenseVector
  }

  def calcKxxDiagj(i: Int, j: Int): DenseVector[Double] = {
    val kXXDiag = covDiag(x, model.g(j).covFunc, model.g(j).covFuncParams)
    kXXDiag(yIdx(i)).toDenseVector
  }
  /**
   * @param k k-derivative
   */
  def calcdKxxDiagj(i: Int, j: Int, k: Int): DenseVector[Double] = {
    val dKxxDiagArray = covDiagD(x, model.g(j).covFunc, model.g(j).covFuncParams)
    val dKxxDiag = dKxxDiagArray(k)
    dKxxDiag(yIdx(i)).toDenseVector
  }

  def calcdKxxDiagi(i: Int, k: Int): DenseVector[Double] = {
    val dKxxDiagArray = covDiagD(x, model.h(i).covFunc, model.h(i).covFuncParams)
    val dKxxDiag = dKxxDiagArray(k)
    dKxxDiag(yIdx(i)).toDenseVector
  }

  def calcAi(i: Int): DenseMatrix[Double] = kXZi(i) * kZZiInv(i)

  def calcdKxzj(i: Int, j: Int, k: Int): DenseMatrix[Double] = {
    val z = model.g(j).z
    val kXZDArray = model.g(j).covFunc.covD(x, z, model.g(j).covFuncParams)
    val dKxz = kXZDArray(k)(yIdx(i), ::).toDenseMatrix
    dKxz
  }

  def calcdKxzi(i: Int, k: Int): DenseMatrix[Double] = {

    val z = model.h(i).z
    val kXZDArray = model.h(i).covFunc.covD(x, z, model.h(i).covFuncParams)
    val dKxz = kXZDArray(k)(yIdx(i), ::).toDenseMatrix
    dKxz
  }

  def calcAj(i: Int, j: Int): DenseMatrix[Double] = kXZj(i, j) * kZZjInv(j)

  def calcdAj(i: Int, j: Int, k: Int): DenseMatrix[Double] = {

    val z = model.g(j).z
    val kZZdArray = model.g(j).covFunc.covD(z, z, model.g(j).covFuncParams)
    val kXZDArray = model.g(j).covFunc.covD(x, z, model.g(j).covFuncParams)
    val kZZinv = kZZjInv(j)

    val dKxz = kXZDArray(k)(yIdx(i), ::).toDenseMatrix

    val Aj = calcAj(i, j)
    val AjD = dKxz * kZZinv - Aj * kZZdArray(k) * kZZinv

    AjD
  }

  def calcdAi(i: Int, k: Int): DenseMatrix[Double] = {

    val z = model.h(i).z
    val kXZDArray = model.h(i).covFunc.covD(x, z, model.h(i).covFuncParams)
    val kZZdArray = model.h(i).covFunc.covD(z, z, model.h(i).covFuncParams)

    val dKxz = kXZDArray(k)(yIdx(i), ::).toDenseMatrix

    val kZZinv = kZZiInv(i)
    val Ai = calcAi(i)

    val dAi = dKxz * kZZinv - Ai * kZZdArray(k) * kZZinv

    dAi
  }
  
  def yi(i: Int): DenseVector[Double] = {
    val y_i = y(::, i)
    y_i(yIdx(i)).toDenseVector
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