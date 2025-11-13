# Use a minimal Python base image
FROM python:alpine
LABEL org.opencontainers.image.source=https://github.com/mcowger/priceServer

# Set the working directory
WORKDIR /app

COPY  prices.py .

# Make the script executable
RUN chmod +x prices.py

# Expose the port the server runs on
EXPOSE 8080

# Run the server on startup
CMD ["python", "prices.py"]
