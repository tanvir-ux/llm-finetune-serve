# CPU image for the OpenAI-compatible server. Compatible with Hugging Face Docker Spaces
# (port 7860, non-root user). The base model is downloaded from the Hub on the first chat request.
FROM python:3.12-slim

RUN useradd -m -u 1000 app
WORKDIR /home/app/llm-finetune-serve
ENV PATH="/home/app/.local/bin:${PATH}" PYTHONUNBUFFERED=1 HF_HOME=/home/app/.cache/huggingface

COPY --chown=app requirements.txt ./
USER app
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app pyproject.toml README.md LICENSE ./
COPY --chown=app src ./src
COPY --chown=app models ./models
RUN pip install --no-cache-dir --no-deps .

EXPOSE 7860
CMD ["llmft", "serve", "--host", "0.0.0.0", "--port", "7860", "--adapter", "models/triage-lora"]
