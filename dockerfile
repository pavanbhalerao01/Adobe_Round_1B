FROM python:3.12.6

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app

COPY . /app

RUN python ./model/download_model.py

CMD ["python", "main.py", "./input"]
