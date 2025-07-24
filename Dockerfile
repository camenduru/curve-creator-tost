FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PYTHONDONTWRITEBYTECODE=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run -d /content -o cuda_12.9.1_575.57.08_linux.run && sh cuda_12.9.1_575.57.08_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home
    
USER camenduru

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 && \
    pip install xformers --extra-index-url https://download.pytorch.org/whl/cu128 && \
    pip install transformers git+https://github.com/huggingface/diffusers.git accelerate sentencepiece protobuf bitsandbytes peft runpod && \    
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/scheduler/scheduler_config.json -d /content/model/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/text_encoder/config.json -d /content/model/text_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/resolve/main/text_encoder/model.safetensors -d /content/model/text_encoder -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/text_encoder_2/config.json -d /content/model/text_encoder_2 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/resolve/main/text_encoder_2/model-00001-of-00002.safetensors -d /content/model/text_encoder_2 -o model-00001-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/resolve/main/text_encoder_2/model-00002-of-00002.safetensors -d /content/model/text_encoder_2 -o model-00002-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/text_encoder_2/model.safetensors.index.json -d /content/model/text_encoder_2 -o model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/tokenizer/merges.txt -d /content/model/tokenizer -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/tokenizer/special_tokens_map.json -d /content/model/tokenizer -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/tokenizer/tokenizer_config.json -d /content/model/tokenizer -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/tokenizer/vocab.json -d /content/model/tokenizer -o vocab.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/tokenizer_2/special_tokens_map.json -d /content/model/tokenizer_2 -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/resolve/main/tokenizer_2/spiece.model -d /content/model/tokenizer_2 -o spiece.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/tokenizer_2/tokenizer.json -d /content/model/tokenizer_2 -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/tokenizer_2/tokenizer_config.json -d /content/model/tokenizer_2 -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/transformer/config.json -d /content/model/transformer -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/resolve/main/transformer/diffusion_pytorch_model-00001-of-00003.safetensors -d /content/model/transformer -o diffusion_pytorch_model-00001-of-00003.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/resolve/main/transformer/diffusion_pytorch_model-00002-of-00003.safetensors -d /content/model/transformer -o diffusion_pytorch_model-00002-of-00003.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/resolve/main/transformer/diffusion_pytorch_model-00003-of-00003.safetensors -d /content/model/transformer -o diffusion_pytorch_model-00003-of-00003.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/transformer/diffusion_pytorch_model.safetensors.index.json -d /content/model/transformer -o diffusion_pytorch_model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/vae/config.json -d /content/model/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/resolve/main/vae/diffusion_pytorch_model.safetensors -d /content/model/vae -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning/raw/main/model_index.json -d /content/model -o model_index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/wan2.1-14b-lora/resolve/main/enlarge.safetensors -d /content/model -o enlarge.safetensors

COPY ./worker_runpod.py /content/worker_runpod.py
WORKDIR /content
CMD python worker_runpod.py