#!/usr/bin/env bash
set -e

# Проверка, что переменные окружения заданы
if [[ -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" ]]; then
  echo "Ошибка: Не заданы переменные окружения AWS_ACCESS_KEY_ID или AWS_SECRET_ACCESS_KEY"
  exit 1
fi

# Сборка backend-образа с использованием BuildKit и секретов
DOCKER_BUILDKIT=1 docker build \
  --secret id=aws_key,env=AWS_ACCESS_KEY_ID \
  --secret id=aws_secret,env=AWS_SECRET_ACCESS_KEY \
  -t credit_api -f Dockerfile.api .
