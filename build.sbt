lazy val root = (project in file(".")).
  settings(
    name := "bayes-scala-gp",
    organization := "com.github.danielkorzekwa",
    version := "0.1-SNAPSHOT",
    scalaVersion := "2.11.4",
    scalacOptions ++= Seq(
      "-feature",
      "-deprecation",
      "-encoding", "UTF-8",       // yes, this is 2 args
      "-unchecked",
      "-Xfuture"
      //"-Ywarn-unused-import"     // 2.11 only
    ),
    
    // Include only src/main/scala in the compile configuration
    unmanagedSourceDirectories in Compile := (scalaSource in Compile).value :: Nil,

    // Include only src/test/scala in the test configuration
    unmanagedSourceDirectories in Test := (scalaSource in Test).value :: Nil,
    
    libraryDependencies ++= Seq(
      "org.scalanlp" %% "breeze" % "0.11.2",
      // test scoped
      "org.slf4j" % "slf4j-log4j12" % "1.7.2" % Test,
      "com.novocode" % "junit-interface" % "0.11" % Test,
       "org.scalanlp" %% "breeze-viz" % "0.10" % Test
      
    )
  )
