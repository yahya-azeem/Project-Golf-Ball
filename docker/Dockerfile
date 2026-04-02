# =============================================================================
# Project Golf — Highly Optimized H100 Training Environment
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Remove interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Install system dependencies (Minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip git ssh openssh-server \
    curl vim htop nvtop ca-certificates ninja-build cmake psmisc build-essential jq \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Update alternatives for python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --set python3 /usr/bin/python3.11

# 3. Upgrade pip and install build tools
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging ninja

# 4. Install core machine learning stack (Minimal - Core Torch Only)
RUN python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# 5. Install training dependencies
RUN python3 -m pip install --no-cache-dir numpy sentencepiece tiktoken huggingface-hub datasets tqdm wandb zstandard psutil

# 6. Install FlashAttention (Requires psutil for metadata generation)
RUN python3 -m pip install --no-cache-dir flash-attn --no-build-isolation

# 7. Setup SSH for RunPod
RUN mkdir /var/run/sshd \
    && echo 'root:root' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# 8. Fix PAM (Resilient version)
RUN if [ -f /etc/pam.d/sshd ]; then \
    sed -i 's/session.*required.*pam_loginuid.so/session optional pam_loginuid.so/g' /etc/pam.d/sshd; \
    fi

# Project directory
WORKDIR /workspace

# Default entrypoint
CMD ["/usr/sbin/sshd", "-D"]
