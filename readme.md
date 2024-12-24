# To Run: 
* Make sure the CUDA tools and driver are installed 

## To run locally:
1. Build the venv & install the python requirements:
> ./Scripts/RebuildVenvLx.sh
2. Run the app:
> uvicorn LLM-Host:app --host 0.0.0.0 --port 41443 --reload

## To run in Docker:
1. Build the Docker image:
> docker build -t [[IMAGENAME]] .
2. Run the Docker container:
> docker run -it --rm --gpus all -p 41443:41443 [[IMAGENAME]]

# To add a new model:
1. Add a new handler file in the LLM/ directory
