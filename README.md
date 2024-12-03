# Resumo

Esse projeto contém uma implementação do PIME_PPO (descrito [nesse paper](https://arxiv.org/abs/2304.10277)), 3 ambientes set point que extendem a classe `base_set_point_env`, 3 modelos de simulações (funções) que encapsulam a lógica de um step no ambiente.

## Modelos de simulação implementados

- *Double water tank*
- *Residual ph water treatment*
- Equipamento CPAP (*Continuous Positive Airway Pressure*)

## Detalhes do ambiente
- A observação é um vetor x (x_1, ..., x_n).
- A ação é um float.
- Reset escolhe os valores do vetor x aleatoriamente.
- Step executa simulation model, calcula o erro e verifica se deve terminar o episódio.

## Como executar o projeto ou scripts relevantes

```bash
# (optional) Create virtual env
python -m venv tcc_venv

# (optional) Activate virtual env (type deactivate to exit)
source tcc_venv/bin/activate  # Linux, macOS
# tcc_venv\Scripts\activate # CMD, Powershell
# source tcc_venv/Scripts/activate # Git Bash terminal open in Windows

# Setup and run training
pip install -r requirements.txt
python src/main.py

# See log results after training
tensorboard --logdir=logs/ppo
```
