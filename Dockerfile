FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "my_env", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"


EXPOSE 8501


# The code to run when container is started:
COPY Dashboard.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "my_env", "python", "Dashboard.py"]