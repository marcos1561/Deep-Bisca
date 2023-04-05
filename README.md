# Deep-Bisca
Projeto para criar e treinar um modelo a jogar bisca, utilizando deep Q-Learning. Também existe uma interface visual básica para jogar contra o modelo (Leia [bisca_game](/bisca_game/README.md) para mais informações).

# Descrição dos módulos
- `bisca_env.py`  
  Contém o ambiente que simula o jogo da bisca. 

- `bisca_components.py`    
  Classes fundamentais para a bisca:
  - `Suit`: Nomes para as representações internas dos diferentes naipes da bisca.
  - `Card`: Representa uma carta da bisca.
  - `Deck`: Representa o baralho da bisca.

- `observation.py`  
  Sistemas que convertem o estado do jogo da bisca para o input a ser utilizado na rede neural.
  Atualmente existem dois sistemas:
  - `Observation`: Sistema base que contém as cartas na mesa, as cartas da mão e a bisca.
  - `History`: Contém tudo que `Observation` tem, mais uma memória de longo prazo das cartas já jogadas que valem ponto.

- `rewards.py`  
  Sistema que controla como as recompensas são dadas de acordo com as ações tomadas em dado estado.

- `players.py`  
  Jogadores da bisca. Existem dois tipos de jogadores:
  - Jogadores internos:  
  Jogadores que estão dentro do ambiente da bisca e vem o jogo de acordo como esse ambiente controla o seu estado.  
  Esses jogadores devem possuir uma forma de escolher uma carta de sua mão para ser jogada, de acordo com o estado atual do jogo.
 
  - Jogadores externos:   
  Jogadores que estão fora do ambiente da bisca e veem o jogo de acordo com algum sistema de `observations.py`.  
  Devem possuir um método para retornar a ação a ser tomada no ambiente, de acordo com o estado dado.  
  
- `player_hand.py`    
  Sistema que gerencia a mão de um jogador interno da bisca.
 
- `models.py`  
  Contém a classe que cria e treina os modelos.
  
- `debug_model.py`  
  Sistemas para analisar como o modelo está se comportando. Atualmente existem dois sistemas:  
    1. Fazer o modelo jogar contro outro jogador de bisca, para ver se, com o avanço do treinamento, o modelo começa a ficar melhor e ganhar mais partidas.
    2. Ver os q-values dados pelo modelo, para estados pré definidos.
