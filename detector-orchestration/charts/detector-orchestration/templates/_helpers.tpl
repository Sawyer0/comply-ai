{{/*
Expand the name of the chart.
*/}}
{{- define "detector-orchestration.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "detector-orchestration.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "detector-orchestration.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "detector-orchestration.labels" -}}
helm.sh/chart: {{ include "detector-orchestration.chart" . }}
{{ include "detector-orchestration.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "detector-orchestration.selectorLabels" -}}
app.kubernetes.io/name: {{ include "detector-orchestration.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "detector-orchestration.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "detector-orchestration.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- required "A valid .Values.serviceAccount.name entry required!" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Redis host configuration
*/}}
{{- define "detector-orchestration.redis.host" -}}
{{- if .Values.redis.enabled }}
{{- if .Values.redis.host }}
{{- .Values.redis.host }}
{{- else }}
{{- printf "%s-redis" (include "detector-orchestration.fullname" .) }}
{{- end }}
{{- end }}
{{- end }}

{{/*
OPA host configuration
*/}}
{{- define "detector-orchestration.opa.host" -}}
{{- if .Values.opa.enabled }}
{{- if .Values.opa.host }}
{{- .Values.opa.host }}
{{- else }}
{{- printf "%s-opa" (include "detector-orchestration.fullname" .) }}
{{- end }}
{{- end }}
{{- end }}
