# ── Frontend Dockerfile ───────────────────────────────────────────────────────
# Stage 1: Build the React/Vite app
# Stage 2: Serve the static build with nginx

# ── Stage 1: Node builder ─────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first for better layer caching
COPY package.json package-lock.json ./

RUN npm ci --ignore-scripts

# Copy the rest of the frontend source
COPY . .

# VITE_API_BASE_URL is intentionally left empty here.
# In production the browser talks to the same origin as the frontend (nginx),
# and nginx proxies /ingest, /query, /health, /uploads → backend:8000.
# If you need a different base URL (e.g. public backend domain), pass it:
#   docker build --build-arg VITE_API_BASE_URL=https://api.example.com
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build


# ── Stage 2: nginx runtime ────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

# Remove the default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the Vite build output
COPY --from=builder /app/dist /usr/share/nginx/html

# nginx listens on port 80
EXPOSE 80

# Healthcheck: ensure nginx is serving
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
