FROM node:16-alpine

# Create app directory
WORKDIR /usr/src/app

# Install backend dependencies
COPY package*.json ./
RUN npm install

# Bundle app source
COPY . .

# Port to access backend
EXPOSE 4300

# PM2 is used to run apps
RUN npm install -g pm2

# define the command to run your app using CMD which defines your runtime
CMD ["pm2-runtime", "ecosystem.config.js"]
