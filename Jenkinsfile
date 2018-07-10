pipeline {
  agent any
  stages {
    stage('Test') {
      steps {
        echo 'Running Tests'
        sh '''#!/bin/bash
source /opt/anaconda3/bin/activate
nosetests tests'''
      }
    }
  }
}