package dk.gp.gpc.factorgraph2

case class BernVariable(p:Double) extends Variable[Double] {

  def calcVariable(): Double = p
}