# Federated Learning Simulation using Flower

This project was developed as part of the course **BITS F464 Machine Learning** course instructed by [Dr. Navneet Goyal](https://www.bits-pilani.ac.in/pilani/navneet-goyal), Senior Professor at BITS Pilani, Rajasthan.

---
## About this Project

This project aims to simulate the federated learning approach by simulating number of clients using [Flower](https://flower.ai/). We simulated multiple clients--representing mobile devices--participating in decentralized model training. We also provided a comparative analysis between federated learning and traditional centralized learning, highlighting performance, efficiency, and accuracy differences.

The goal is to mimic real-world federated scenarios where client availability and capability can vary, such as on mobile devices with limited battery or processing power.

#### Key Features :

- Client Simulation using Flower's local simulation backend.
- Client Metadata Generation to simulate device-specific properties.
- Custom Client Selection Strategy based on simulated availability.
- Federated vs. Centralized Training comparison on model performance.
- Visualizations of label distribution and client availability across rounds.
- Dirichlet-based Partitioning for non-IID data simulation.

## How It Works

#### Client Metadata
Each client simulates a mobile device, characterized by:
- Battery Power: Determines if the device has enough charge to participate.
- Processor Support: Whether the processor supports model training.

At the start of each round, only clients with sufficient power and processor support are allowed to participate.

#### Dataset and Partitioning
- We used the [huggan/AFHQ](https://huggingface.co/datasets/huggan/AFHQ) dataset and partitioned it using:
- Dirichlet Partitioning: Enables non-IID distribution across clients by varying alpha.
- Visualization Support: Display how labels are distributed across partitions.

#### Centralized vs Federated Comparison
We trained the same model:
- Once using federated learning (different datasets sent to each client and weights from the trained model of each client are aggregated).
- Once in a centralized setting (all data pooled and a single model is trained with all the data).

#### The performance was compared across:
- Loss (distributed and centralized evaluation).
- Accuracy (client-reported and central evaluation).

## What's Inside

| File/Folder | Description |
| ---- | ---------- |
| [cml-simulation](cml-simulation/) | Files for simulating Centralized Learning Approach |
| [fl-simulation/](fl-simulation/) | Files for simulating Federated Learning Approach |

---
## Cloning from github
```
git clone https://github.com/Saarthak-Vijayvargiya-github/Federated-Learning-Sim.git
cd Federated-Learning-Sim
```

---
## Have fun! Thank-You!