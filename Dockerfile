FROM python:3.10.0

WORKDIR .

COPY requirements.txt ./
ADD ./data ./data
ADD ./weights ./weights
COPY ./llm_script.py ./

RUN pip install --no-cache-dir -r requirements.txt


CMD [ "uvicorn", "llm_script:app", "--host", "0.0.0.0", "--port", "8080" ]
