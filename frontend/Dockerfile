# Multi-stage Dockerfile for building the frontend and serving with nginx

# Stage 1: build the app with node
FROM node:20-alpine AS builder
WORKDIR /app

# Copy package manifests and install dependencies first (cache)
COPY package.json package-lock.json* pnpm-lock.yaml* ./

# Use pnpm if lockfile present, otherwise npm
RUN if [ -f pnpm-lock.yaml ]; then \
      npm install -g pnpm && pnpm install --frozen-lockfile; \
    elif [ -f package-lock.json ]; then \
      npm ci; \
    else \
      # no lockfile present in the repo; fall back to a regular install
      npm install; \
    fi

# Copy source
COPY . .

# Build
RUN npm run build

# Stage 2: serve with nginx
FROM nginx:stable-alpine

# Remove default nginx website
RUN rm -rf /usr/share/nginx/html/*

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
