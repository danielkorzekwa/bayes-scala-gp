package dk.gp.util

import java.io.FileOutputStream
import com.sun.org.apache.xalan.internal.xsltc.compiler.Output
import com.twitter.chill.ScalaKryoInstantiator
import com.esotericsoftware.kryo.io.Output

object saveObject {
  
  def apply[T](obj: T, file: String) = {

    val instantiator = new ScalaKryoInstantiator()
    instantiator.setRegistrationRequired(false)

    val kryo = instantiator.newKryo()
    val fileOut = new FileOutputStream(file)
    val output = new Output(fileOut)
    kryo.writeObject(output, obj)
    output.close()
  }
}