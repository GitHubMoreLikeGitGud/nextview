# Base image pinned by digest (manifest list for public.ecr.aws/lambda/python:3.12, resolved 2026-10-23;
# linux/amd64 manifest sha256:c73d459e2dcaefa83c47c095560ad164d732251312b3371c7a49c46f1c4d4ff5).
FROM public.ecr.aws/lambda/python:3.12@sha256:25aee9bac65f131c5d090d3e3a36ccf232d65cf3e6c1dbd0bffd43ca7ce34406

COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt --target ${LAMBDA_TASK_ROOT}

COPY config ${LAMBDA_TASK_ROOT}/config
COPY src/nextview ${LAMBDA_TASK_ROOT}/nextview

ARG GIT_SHA=dev
RUN printf 'GIT_SHA = "%s"\n' "${GIT_SHA}" > ${LAMBDA_TASK_ROOT}/nextview/_build.py

ENV NEXTVIEW_CONFIG_DIR=${LAMBDA_TASK_ROOT}/config \
    OPENBLAS_NUM_THREADS=1

CMD ["nextview.lambda_app.handler.handler"]
