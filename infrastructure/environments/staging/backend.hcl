bucket = "mlops-tfstate-bucket-roman-1766514292"  # пример
key    = "staging/terraform.tfstate"
region = "ru-central1"

endpoints = {
  s3 = "https://storage.yandexcloud.net"
}

skip_region_validation      = true
skip_credentials_validation = true
skip_metadata_api_check     = true
skip_requesting_account_id  = true
skip_s3_checksum            = true

use_path_style = true
