variable "cloud_id" { type = string }
variable "folder_id" { type = string }
variable "zone" {
  type    = string
  default = "ru-central1-a"
}

variable "name" {
  type    = string
  default = "mlops2"
}

variable "cidr_vpc" {
  type    = string
  default = "10.10.0.0/16"
}
variable "cidr_subnet" {
  type    = string
  default = "10.10.10.0/24"
}
variable "k8s_api_cidrs" {
  type    = list(string)
  default = ["0.0.0.0/0"]
}

variable "state_bucket" { type = string }
variable "artifacts_bucket" { type = string }

variable "k8s_version" {
  type    = string
  default = "1.31"
}

variable "cpu_platform_id" {
  type    = string
  default = "standard-v3"
}
variable "gpu_platform_id" {
  type    = string
  default = "gpu-standard-v3"
}
