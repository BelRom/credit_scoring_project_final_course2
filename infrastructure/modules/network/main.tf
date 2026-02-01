resource "yandex_vpc_network" "this" {
  name      = "${var.name}-net"
  folder_id = var.folder_id
}

resource "yandex_vpc_subnet" "this" {
  name           = "${var.name}-subnet"
  folder_id      = var.folder_id
  zone           = var.zone
  network_id     = yandex_vpc_network.this.id
  v4_cidr_blocks = [var.cidr_subnet]
}

# SG для Managed Kubernetes control-plane / nodes
resource "yandex_vpc_security_group" "k8s_nodes" {
  name       = "${var.name}-k8s-nodes-sg"
  folder_id  = var.folder_id
  network_id = yandex_vpc_network.this.id

  # Между нодами (pod-to-pod / node-to-node)
  ingress {
    protocol       = "ANY"
    description    = "Node-to-node"
    v4_cidr_blocks = [var.cidr_vpc]
    from_port      = 0
    to_port        = 65535
  }

  egress {
    protocol       = "ANY"
    description    = "Outbound any"
    v4_cidr_blocks = ["0.0.0.0/0"]
    from_port      = 0
    to_port        = 65535
  }

  # Доступ к ingress-nginx по NodePort (для демо)
  ingress {
    protocol       = "TCP"
    description    = "Ingress HTTP via NodePort"
    v4_cidr_blocks = ["0.0.0.0/0"]
    port           = 31351
  }

  ingress {
    protocol       = "TCP"
    description    = "Ingress HTTPS via NodePort"
    v4_cidr_blocks = ["0.0.0.0/0"]
    port           = 30843
  }
}

resource "yandex_vpc_security_group" "k8s_api" {
  name       = "${var.name}-k8s-api-sg"
  folder_id  = var.folder_id
  network_id = yandex_vpc_network.this.id

  ingress {
    protocol       = "TCP"
    description    = "K8s API access"
    v4_cidr_blocks = var.k8s_api_cidrs
    port           = 443
  }
}
