install:
	pip install	--upgrade pip && \
	pip install -r requirements.txt

format: 
	black app.py
lint:
	pylint --fail-under=-1 app.py
