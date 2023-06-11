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
        stage('Build and Push Docker Image') {
            steps {
                // Build Docker image
                sh 'docker build -t FYPDEEPPD005:latest -f Dockerfile .'
                
                // Authenticate with Docker Hub using Jenkins credentials
                withCredentials([usernamePassword(credentialsId: 'b18fb15d-147f-439c-add3-13ce3b2757c1', usernameVariable: 'DOCKERHUB_USERNAME', passwordVariable: 'DOCKERHUB_PASSWORD')]) {
                    // Login to Docker Hub
                    sh 'docker login -u $DOCKERHUB_USERNAME -p $DOCKERHUB_PASSWORD'
                    
                    // Push Docker image to Docker Hub
                    sh 'docker push FYPDEEPPD005:latest'
                }
            }
        }
    }
}
