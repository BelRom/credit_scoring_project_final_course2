variable "folder_id" { type = string }
variable "name" { type = string }
variable "zone" {
  type    = string
  default = "ru-central1-a"
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
  description = "CIDR-ы, которым разрешён доступ к Kubernetes API"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
