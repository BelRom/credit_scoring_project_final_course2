variable "folder_id" { type = string }
variable "name" { type = string }
variable "zone" {
  type    = string
  default = "ru-central1-a"
}

variable "network_id" { type = string }
variable "subnet_id" { type = string }

variable "sg_nodes_id" { type = string }
variable "sg_api_id" { type = string }

variable "k8s_version" {
  type    = string
  default = "1.31"
}

# CPU node group
variable "cpu_cores" {
  type    = number
  default = 4
}
variable "cpu_memory" {
  type    = number
  default = 8
}
variable "cpu_disk_gb" {
  type    = number
  default = 50
}
variable "cpu_min" {
  type    = number
  default = 1
}
variable "cpu_max" {
  type    = number
  default = 3
}

# GPU node group

variable "enable_gpu" {
  type    = bool
  default = false
}
variable "gpu_cores" {
  type    = number
  default = 8
}
variable "gpu_memory" {
  type    = number
  default = 32
}
variable "gpu_disk_gb" {
  type    = number
  default = 100
}
variable "gpu_min" {
  type    = number
  default = 0
}
variable "gpu_max" {
  type    = number
  default = 1
}

# platform_id 
variable "cpu_platform_id" {
  type    = string
  default = "standard-v3"
}
variable "gpu_platform_id" {
  type    = string
  default = "gpu-standard-v3"
}

variable "cluster_name" {
  type        = string
  description = "Name of Kubernetes cluster"
}
