pipeline {
  agent any
  stages {
    stage('Test') {
      steps {
        echo 'Running Tests'
        sh 'nosetests tests'
      }
    }
  }
}