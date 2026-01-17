resource "yandex_storage_bucket" "state" {
  bucket    = var.state_bucket
  folder_id = var.folder_id

  versioning {
    enabled = true
  }
}

resource "yandex_storage_bucket" "artifacts" {
  bucket    = var.artifacts_bucket
  folder_id = var.folder_id

  versioning {
    enabled = true
  }
}
