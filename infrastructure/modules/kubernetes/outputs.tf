output "cluster_id" { value = yandex_kubernetes_cluster.this.id }
output "cluster_name" { value = yandex_kubernetes_cluster.this.name }
output "cluster_ca" {
  value     = yandex_kubernetes_cluster.this.master[0].cluster_ca_certificate
  sensitive = true
}
output "endpoint" { value = yandex_kubernetes_cluster.this.master[0].external_v4_endpoint }
