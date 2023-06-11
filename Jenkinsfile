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
        stage('Build and Push Docker Image') {
            steps {
                // Build Docker image
                bat 'docker build -t FYPDeepPD:latest -f Dockerfile .'
                
                // Authenticate with Docker Hub using Jenkins credentials
                withCredentials([usernamePassword(credentialsId: 'b18fb15d-147f-439c-add3-13ce3b2757c1', usernameVariable: 'DOCKERHUB_USERNAME', passwordVariable: 'DOCKERHUB_PASSWORD')]) {
                    // Login to Docker Hub
                    bat 'docker login -u $DOCKERHUB_USERNAME -p $DOCKERHUB_PASSWORD'
                    
                    // Push Docker image to Docker Hub
                    bat 'docker push FYPDeepPD:latest'
                }
            }
        }
        
    }
}
