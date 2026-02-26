# Azure Container Apps Deploy (RiskSentinel)

This guide deploys the existing Streamlit app to Azure Container Apps.

## 1. Prerequisites

- Azure CLI logged in: `az login`
- Subscription selected: `az account set --subscription "<SUBSCRIPTION_NAME_OR_ID>"`
- Azure OpenAI resource/deployments already created
- Dataset available in a mounted path at runtime (see note below)

## 2. Variables

```bash
RG="rg-risksentinel-dev"
LOCATION="swedencentral"
ACR_NAME="acrrisksentinel$RANDOM"
ACA_ENV="acaenv-risksentinel"
ACA_APP="risksentinel-app"
IMAGE_NAME="risksentinel:latest"
```

## 3. Create infrastructure

```bash
az group create -n "$RG" -l "$LOCATION"
az acr create -g "$RG" -n "$ACR_NAME" --sku Basic
az acr login -n "$ACR_NAME"
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az containerapp env create -g "$RG" -n "$ACA_ENV" -l "$LOCATION"
```

## 4. Build and push image

```bash
ACR_LOGIN_SERVER=$(az acr show -g "$RG" -n "$ACR_NAME" --query loginServer -o tsv)
docker build -t "$ACR_LOGIN_SERVER/$IMAGE_NAME" .
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME"
```

## 5. Create Container App

```bash
az containerapp create \
  -g "$RG" \
  -n "$ACA_APP" \
  --environment "$ACA_ENV" \
  --image "$ACR_LOGIN_SERVER/$IMAGE_NAME" \
  --target-port 8501 \
  --ingress external \
  --registry-server "$ACR_LOGIN_SERVER" \
  --cpu 1.0 \
  --memory 2.0Gi \
  --min-replicas 0 \
  --max-replicas 1
```

## 6. Configure secrets and env vars

```bash
az containerapp secret set \
  -g "$RG" -n "$ACA_APP" \
  --secrets \
  azure-openai-api-key="<YOUR_AZURE_OPENAI_API_KEY>"

az containerapp update \
  -g "$RG" -n "$ACA_APP" \
  --set-env-vars \
  AZURE_OPENAI_ENDPOINT="https://swedencentral.api.cognitive.microsoft.com/" \
  AZURE_OPENAI_DEPLOYMENT="gpt-4o" \
  AZURE_OPENAI_FALLBACK_DEPLOYMENT="gpt-4o-mini" \
  AZURE_OPENAI_API_VERSION="2025-03-01-preview" \
  AZURE_OPENAI_API_KEY=secretref:azure-openai-api-key \
  JUDGE_ACCESS_CODE="<JUDGE_CODE>" \
  GPT_MAX_CALLS_PER_MINUTE_SESSION="8" \
  GPT_MAX_CALLS_PER_MINUTE_GLOBAL="20" \
  GPT_MAX_CALLS_PER_SESSION="120" \
  RISKSENTINEL_DATA_ROOT="/mnt/risksentinel-data"
```

## 7. Get public URL

```bash
az containerapp show -g "$RG" -n "$ACA_APP" --query properties.configuration.ingress.fqdn -o tsv
```

Open: `https://<fqdn>`

## Dataset note (important)

The repo does not include the full processed dataset. The app needs:
- `sector_mapping.parquet`
- `networks/node_centralities.pkl`
- plus the other processed files used by the loaders

Set `RISKSENTINEL_DATA_ROOT` to a path containing processed data.

Recommended for Azure:
- Mount Azure Files / BlobFuse to `/mnt/risksentinel-data`
- Keep only one read-only shared dataset for all revisions

## Post-hackathon reliability

The app keeps working after March 2026 as long as:
- Azure subscription remains active
- Azure OpenAI quota/keys remain valid
- Container App and dataset mount stay online
- you keep secrets and model deployments configured
