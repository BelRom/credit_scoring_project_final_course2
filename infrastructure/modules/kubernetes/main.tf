############################################
# Service Accounts
############################################

resource "yandex_iam_service_account" "k8s_sa" {
  name      = "${var.name}-k8s-sa"
  folder_id = var.folder_id
}

resource "yandex_iam_service_account" "k8s_nodes_sa" {
  name      = "${var.name}-k8s-nodes-sa"
  folder_id = var.folder_id
}

############################################
# IAM Bindings
############################################

resource "yandex_resourcemanager_folder_iam_binding" "k8s_sa_admin" {
  folder_id = var.folder_id
  role      = "k8s.clusters.agent"
  members   = ["serviceAccount:${yandex_iam_service_account.k8s_sa.id}"]
}

resource "yandex_resourcemanager_folder_iam_binding" "k8s_nodes_sa_pull" {
  folder_id = var.folder_id
  role      = "container-registry.images.puller"
  members   = ["serviceAccount:${yandex_iam_service_account.k8s_nodes_sa.id}"]
}

resource "yandex_resourcemanager_folder_iam_binding" "k8s_nodes_sa_editor" {
  folder_id = var.folder_id
  role      = "editor"
  members   = ["serviceAccount:${yandex_iam_service_account.k8s_nodes_sa.id}"]
}

resource "yandex_resourcemanager_folder_iam_binding" "k8s_sa_vpc_public_admin" {
  folder_id = var.folder_id
  role      = "vpc.publicAdmin"
  members   = ["serviceAccount:${yandex_iam_service_account.k8s_sa.id}"]
}

############################################
# Managed Kubernetes Cluster
############################################

resource "yandex_kubernetes_cluster" "this" {
  name       = "${var.name}-k8s"
  folder_id  = var.folder_id
  network_id = var.network_id

  cluster_ipv4_range = "10.112.0.0/16"
  service_ipv4_range = "10.96.0.0/16"

  master {
    version = var.k8s_version

    zonal {
      zone      = var.zone
      subnet_id = var.subnet_id
    }

    public_ip          = true
    security_group_ids = [var.sg_api_id, var.sg_nodes_id]
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes = [
      master,
      release_channel,
      labels,
      cluster_ipv4_range,
      service_ipv4_range,
    ]
  }

  service_account_id      = yandex_iam_service_account.k8s_sa.id
  node_service_account_id = yandex_iam_service_account.k8s_nodes_sa.id
}

############################################
# CPU node group
############################################

data "yandex_kubernetes_cluster" "existing" {
  name      = "${var.name}-k8s"
  folder_id = var.folder_id
}

resource "yandex_kubernetes_node_group" "cpu" {
  name       = "${var.name}-cpu-ng"
  cluster_id = data.yandex_kubernetes_cluster.existing.id
  version    = var.k8s_version

  instance_template {
    platform_id = var.cpu_platform_id

    resources {
      cores  = var.cpu_cores
      memory = var.cpu_memory
    }

    boot_disk {
      type = "network-ssd"
      size = var.cpu_disk_gb
    }

    network_interface {
      subnet_ids         = [var.subnet_id]
      security_group_ids = [var.sg_nodes_id]
      nat                = true
    }

    scheduling_policy {
      preemptible = true
    }
  }

  scale_policy {
    auto_scale {
      min     = var.cpu_min
      max     = var.cpu_max
      initial = var.cpu_min
    }
  }

  allocation_policy {
    location {
      zone = var.zone
    }
  }

  depends_on = [
    yandex_resourcemanager_folder_iam_binding.k8s_nodes_sa_pull,
    yandex_resourcemanager_folder_iam_binding.k8s_nodes_sa_editor,
    yandex_resourcemanager_folder_iam_binding.k8s_sa_admin,
    yandex_resourcemanager_folder_iam_binding.k8s_sa_vpc_public_admin,
  ]

  timeouts {
    create = "40m"
    update = "40m"
    delete = "40m"
  }
}

############################################
# GPU node group
############################################

resource "yandex_kubernetes_node_group" "gpu" {
  count      = var.enable_gpu ? 1 : 0
  name       = "${var.name}-gpu-ng"
  cluster_id = yandex_kubernetes_cluster.this.id
  version    = var.k8s_version

  instance_template {
    platform_id = "gpu-standard-v3"

    resources {
      cores         = 8
      memory        = 32
      gpus          = 1
      core_fraction = 100
    }

    boot_disk {
      type = "network-ssd"
      size = 64
    }

    network_interface {
      subnet_ids         = [var.subnet_id]
      security_group_ids = [var.sg_nodes_id]
      nat                = true
    }

    # Taints задаём через kubelet metadata (рабочий способ для MKS)
    metadata = {
      "kubelet_extra_args" = "--register-with-taints=nvidia.com/gpu=true:NoSchedule"
    }

    container_runtime {
      type = "containerd"
    }
  }

  scale_policy {
    fixed_scale {
      size = 1
    }
  }

  allocation_policy {
    location {
      zone = var.zone
    }
  }
}
