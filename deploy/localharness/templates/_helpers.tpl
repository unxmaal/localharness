{{- define "localharness.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "localharness.labels" -}}
app.kubernetes.io/name: {{ include "localharness.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "localharness.storeEnv" -}}
- name: LOCALHARNESS_STORE
  value: postgres
- name: PGHOST
  value: {{ .Values.postgres.host | default (printf "%s-postgres" (include "localharness.name" .)) | quote }}
- name: PGDATABASE
  value: {{ .Values.postgres.database | quote }}
- name: PGUSER
  value: {{ .Values.postgres.user | quote }}
- name: PGPASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Values.postgres.existingSecret | default (printf "%s-postgres" (include "localharness.name" .)) | quote }}
      key: password
{{- end -}}
