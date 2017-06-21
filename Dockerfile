FROM python:3.6

# Install pipenv
ADD https://raw.github.com/kennethreitz/pipenv/master/get-pipenv.py /get-pipenv.py
RUN python /get-pipenv.py && rm /get-pipenv.py

# Install Node.js 6.x and yarn
RUN curl -sL https://deb.nodesource.com/setup_6.x | bash -
RUN curl -sL https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - && \
    echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list
RUN apt-get update && apt-get install -y nodejs yarn

# Add source code
COPY . /app
WORKDIR /app

# Install dependencies
RUN pipenv install --system
RUN make init-js build-js

# Run app
EXPOSE 5000
CMD gunicorn main:application --access-logfile - --workers 4 --bind 0.0.0.0:5000
