package dk.gp.gpc.factorgraph2

import dk.bayes.math.gaussian.canonical.CanonicalGaussian

case class GaussianVariable() extends Variable[CanonicalGaussian] {

  def calcVariable(): CanonicalGaussian = {
    val tailProduct =  CanonicalGaussian.multOp(getMessages().tail.map(_.get): _*)
    CanonicalGaussian.multOp(getMessages().head.get,tailProduct)
  }
}