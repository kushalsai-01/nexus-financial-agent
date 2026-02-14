terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "nexus-terraform-state"
    key    = "nexus/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "nexus-financial-agent"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

resource "aws_vpc" "nexus" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = { Name = "nexus-${var.environment}" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.nexus.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = { Name = "nexus-public-${count.index + 1}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.nexus.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = { Name = "nexus-private-${count.index + 1}" }
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_internet_gateway" "nexus" {
  vpc_id = aws_vpc.nexus.id
  tags   = { Name = "nexus-igw" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.nexus.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.nexus.id
  }

  tags = { Name = "nexus-public-rt" }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "nexus_app" {
  name   = "nexus-app-${var.environment}"
  vpc_id = aws_vpc.nexus.id

  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Streamlit dashboard"
  }

  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "Prometheus metrics"
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "nexus-app-sg" }
}

resource "aws_security_group" "nexus_db" {
  name   = "nexus-db-${var.environment}"
  vpc_id = aws_vpc.nexus.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.nexus_app.id]
    description     = "PostgreSQL from app"
  }

  tags = { Name = "nexus-db-sg" }
}

resource "aws_db_subnet_group" "nexus" {
  name       = "nexus-${var.environment}"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_db_instance" "nexus" {
  identifier     = "nexus-${var.environment}"
  engine         = "postgres"
  engine_version = "16.1"
  instance_class = var.db_instance_class

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true

  db_name  = "nexus"
  username = "nexus"
  password = "change-me-in-production"

  db_subnet_group_name   = aws_db_subnet_group.nexus.name
  vpc_security_group_ids = [aws_security_group.nexus_db.id]

  backup_retention_period = 7
  skip_final_snapshot     = var.environment != "prod"

  tags = { Name = "nexus-db" }
}

resource "aws_ecr_repository" "nexus" {
  name                 = "nexus-financial-agent"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

output "vpc_id" {
  value = aws_vpc.nexus.id
}

output "db_endpoint" {
  value = aws_db_instance.nexus.endpoint
}

output "ecr_url" {
  value = aws_ecr_repository.nexus.repository_url
}
