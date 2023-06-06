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
                bat 'python -m pip install --upgrade pip'
                bat 'pip install -r requirements.txt'
                echo 'STARTING Pipeline'
            }
        }
        stage('Format Code with Black') {
            steps {
                // Run Black formatter on app.py
                bat 'black app.py'
                echo 'Black has formatted app.py'
            }
        }
        
        
    }
}
