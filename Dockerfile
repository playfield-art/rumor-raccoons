# Use the official lightweight Python image.
FROM python:3.10-slim
# Allow statements and log
ENV PYTHONUNBUFFERED True
# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
# Install production dependencies.
RUN pip install -r requirements.txt
# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]