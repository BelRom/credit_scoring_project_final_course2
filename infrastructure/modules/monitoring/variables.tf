variable "name" { type = string }

variable "kube_host" { type = string }
variable "kube_ca" { type = string }
variable "kube_token" {
  type      = string
  sensitive = true
}
