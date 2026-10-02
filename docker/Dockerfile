# The H3 SYCL work's build image: the Strata port's oneAPI image plus oneDNN's headers (the runtime is already there).
FROM strata-sycl-dev
RUN apt-get update -qq && apt-get install -y -qq intel-oneapi-dnnl-devel-2026.0 && rm -rf /var/lib/apt/lists/*
