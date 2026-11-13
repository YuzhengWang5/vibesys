ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
COPY copilot_usage.py /opt/copilot_usage.py
RUN mkdir -p /opt/copilot-bin \
    && printf '#!/usr/bin/env bash\nexec /usr/bin/python3 /opt/copilot_usage.py "$@"\n' > /opt/copilot-bin/copilot \
    && chmod 755 /opt/copilot-bin/copilot
ENV PATH="/opt/copilot-bin:${PATH}"
ENV COPILOT_NATIVE_BINARY="/usr/local/bin/copilot"
USER agent
