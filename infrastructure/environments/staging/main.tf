provider "yandex" {
  token     = var.yc_token
  cloud_id  = var.cloud_id
  folder_id = var.folder_id
  zone      = var.zone
}

module "network" {
  source        = "../../modules/network"
  folder_id     = var.folder_id
  name          = var.name
  zone          = var.zone
  cidr_vpc      = var.cidr_vpc
  cidr_subnet   = var.cidr_subnet
  k8s_api_cidrs = var.k8s_api_cidrs
}

module "storage" {
  source           = "../../modules/storage"
  folder_id        = var.folder_id
  name             = var.name
  state_bucket     = var.state_bucket
  artifacts_bucket = var.artifacts_bucket
}

module "kubernetes" {
  source       = "../../modules/kubernetes"
  cluster_name = var.name
  folder_id    = var.folder_id
  name         = var.name
  zone         = var.zone

  network_id  = module.network.network_id
  subnet_id   = module.network.subnet_id
  sg_nodes_id = module.network.sg_nodes_id
  sg_api_id   = module.network.sg_api_id

  k8s_version = var.k8s_version

  cpu_platform_id = var.cpu_platform_id
  gpu_platform_id = var.gpu_platform_id
}
