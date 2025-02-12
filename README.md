# Resumo

Esse projeto contém uma implementação do PIME_PPO (descrito [nesse paper](https://arxiv.org/abs/2304.10277)), 2 ambientes set point que extendem a classe `base_set_point_env`, 2 modelos de simulações (funções / equações diferenciais / funções de transferência) discretizados com o método de Euler e que encapsulam a lógica de um step no ambiente.

## Modelos de simulação implementados

- *Double water tank*
- *Residual ph water treatment* (incomplete, need a formula inside simulation model)
- Equipamento CPAP (*Continuous Positive Airway Pressure*)

## Detalhes do ambiente
- A observação é um vetor x (x_1, ..., x_n) combinado com a dupla (y_ref, z_t).
- A ação é um float.
- Reset escolhe os valores do vetor x aleatoriamente ou inicializa sempre os mesmos valores escolhido por um parâmetro do ambiente.
- Step executa simulation model, atualiza a observação, calcula o erro (y_ref - y_current), recompensa (-(error²)) e verifica se deve terminar o episódio.

## Como executar o projeto ou scripts relevantes

```bash
# (optional) Create virtual env
python -m venv tcc_venv

# (optional) Activate virtual env (type deactivate to exit)
source tcc_venv/bin/activate  # Linux, macOS
# tcc_venv\Scripts\activate # CMD, Powershell
# source tcc_venv/Scripts/activate # Git Bash terminal open in Windows

# Setup and run training (requires Nvidia GPU, edit requirements.txt to toggle for CPU only)
pip install -r requirements.txt
python src/main.py
```
<!-- # See log results after training (log folder is created automatically) -->
<!-- tensorboard --logdir=logs/ppo -->
