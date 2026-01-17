output "network_id" { value = yandex_vpc_network.this.id }
output "subnet_id" { value = yandex_vpc_subnet.this.id }
output "sg_nodes_id" { value = yandex_vpc_security_group.k8s_nodes.id }
output "sg_api_id" { value = yandex_vpc_security_group.k8s_api.id }
