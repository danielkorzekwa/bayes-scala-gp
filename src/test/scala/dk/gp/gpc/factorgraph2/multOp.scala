package dk.gp.gpc.factorgraph2

trait multOp[X] {
  
  def apply(x: X*): X
}