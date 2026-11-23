# AWS deployment (configuration only - not deployed)

This folder holds a ready-to-fill **ECS Fargate** task definition. It was **not** deployed as part of the
internship (no AWS account is required to evaluate the project).

## Steps (when you have an AWS account)
1. Create an ECR repo and push the image:
   ```bash
   aws ecr create-repository --repository-name purchase-intent-api
   docker build -f docker/Dockerfile.api -t purchase-intent-api:1.0.0 .
   aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
   docker tag purchase-intent-api:1.0.0 <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/purchase-intent-api:1.0.0
   docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/purchase-intent-api:1.0.0
   ```
2. Replace `<ACCOUNT_ID>` and `<REGION>` in `ecs-task-definition.json`.
3. Register the task definition and create a Fargate service behind an Application Load Balancer
   (health check path: `/health`).
4. The dashboard can be deployed the same way with `docker/Dockerfile.dashboard` (port 8501).
