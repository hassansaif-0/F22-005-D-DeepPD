pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                // Checkout code from repository
                checkout scm
            }
        }
        stage('Install Dependencies') {
            steps {
                // Install required Python packages
                sh 'python -m pip install --upgrade pip'
                sh 'pip install -r requirements.txt'
                echo 'STARTING Pipeline'
            }
        }
        stage('Format Code with Black') {
            steps {
                // Run Black formatter on app.py
                sh 'black app.py'
                echo 'Black has formatted app.py'
            }
        }
        
        
    }
}
