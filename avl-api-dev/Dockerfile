# Use the official Python image as the base image
FROM python:3.8-slim-buster

# build variables.
ENV DEBIAN_FRONTEND noninteractive

# install Microsoft SQL Server requirements.
ENV ACCEPT_EULA=Y
RUN apt-get update -y && apt-get update \
  && apt-get install -y --no-install-recommends curl gcc g++ gnupg unixodbc-dev

# Add SQL Server ODBC Driver 17 for Ubuntu 18.04
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
  && curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list \
  && apt-get update \
  && apt-get install -y --no-install-recommends --allow-unauthenticated msodbcsql17 mssql-tools \
  && echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bash_profile \
  && echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc

# Set the working directory in the container
WORKDIR C:\app

# Copy the app code and requirements into the container
COPY app.py .
COPY requirements.txt .

# Install app dependencies
RUN pip install -r requirements.txt

# clean the install.
RUN apt-get -y clean

# Expose the port your Flask app will run on (default is 5000)
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]