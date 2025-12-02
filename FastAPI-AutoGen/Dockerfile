FROM python:3.11


WORKDIR /code/app

RUN pip install playwright

RUN playwright install

RUN playwright install-deps


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./app /code/app


CMD ["python3", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "fastapi", "run", "main.py", "--port", "8086"]

