FROM vllm/vllm-openai:v0.10.0
RUN python3 -m pip install --no-cache-dir transformers==4.57.1 \
    'numpy>=1.26,<2' 'pydantic>=2' 'httpx>=0.27' fastapi uvicorn
RUN ln -s /usr/bin/python3 /usr/local/bin/python
ENTRYPOINT []
