# Use a minimal Python base image
FROM python:alpine

# Set the working directory
WORKDIR /app

# Copy the convert_api_to_model_prices.py script and the overrides map
COPY overrides.json convert_api_to_model_prices.py .

# Make the script executable
RUN chmod +x convert_api_to_model_prices.py

# Expose the port the server runs on
EXPOSE 8080

# Run the server on startup
CMD ["python", "convert_api_to_model_prices.py", "--serve", "--input_url", "https://models.dev/api.json", "--port", "8080", "--overrides", "overrides.json"]
