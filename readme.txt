How to Build and Run It with Docker

Step 0: Set up in .env your GROQ_API_KEY= 

Step 1: Build the Image (Choose GPU or CPU)
You need to run the build command once to create your application image.

Target	Command
GPU (NVIDIA with Cuda Toolkit version 12.8)	
docker-compose -f docker-compose.yml build

CPU (Universal)	
docker-compose -f docker-compose-cpu.yml build

Export to Sheets
Step 2: Run the Services
After a successful build, use the same file to start both the app and ollama services.

Target	Command
GPU (NVIDIA)	docker-compose -f docker-compose.yml up
CPU (Universal)	docker-compose -f docker-compose-cpu.yml up

Export to Sheets
The first time you run the up command, the ollama service will automatically pull the qwen2.5:latest model, which will be persisted to the ollama_models volume for faster startups in the future.

Once the services are running, access your application at: http://localhost:8000