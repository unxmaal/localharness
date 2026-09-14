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

{{- define "localharness.githubEnv" -}}
{{/*
  Every tier that reaches GitHub needs this, and the first fan-out proved it
  the hard way: the sweep carried the token, inspect did not, and all 125
  candidates failed with "populate the GH_TOKEN environment variable". The
  sweep still worked because the feeds are public and the API is not.

  Optional on purpose: without a token the public API still answers at a lower
  rate limit, so an unconfigured cluster degrades rather than refuses.
*/}}
- name: GH_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ include "localharness.name" . }}-gh
      key: token
      optional: true
{{- end -}}
