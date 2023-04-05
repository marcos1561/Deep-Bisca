# Bisca Game
Interface visual bem simples para jogar bisca contra os modelos treinados.

# Como abrir o jogo
Caso você não tenho os requisitos necessários. Abra um terminal nessa pasta e execute o seguinte comando

    pip install -r requirements.txt

Para jogar bisca, abra um terminal nessa pasta e execute o seguinte comando :)

    python .\main.py

> OBS: A pasta raíz desse projeto precisa estar no PYTHONPATH para o arquivo `main.py` rodar, pois o mesmo utiliza módulos como `bisca_env.py`, etc. Alternativamente, é possível copiar os arquivos da pasta raíz para esta pasta.

# Como jogar
Assim que abrir o jogo, primeiro uma tela solicitando o seu nome aparecerá:

<img src="/bisca_game/game_images/tela_inicial.png" alt="Tela inicial" width="50%" height="50%">

Após inserir o nome e pressionar enter, o jogo começa. A tela do jogo contém a bisca do jogo, as cartas da sua mão e as cartas jogadas na mesa. A imagem a seguir exemplifica essa tela

<img src="/bisca_game/game_images/bisca_game_view.png" alt="Tela inicial" width="80%" height="80%">

Para jogar, apenas é necessário clicar sobre alguma carta da sua mão para joga-lá na mesa. 

Caso você seja o primeiro a jogar, o modelo treinado joga logo após de você escolher uma carta, caso ele seja o primeiro a jogar, ele imediante joga quando a rodada começa, e sua carta jogaga aparece na mesa.

Em uma determinada rodada, após ter selecionado uma carta, as duas cartas jogadas na mesa ficam aparecendo até você clicar em qualquer lugar na tela para prosseguir para a próxima rodada.

Quando o jogo acabar, no canto esquerdo superior aparece quem ganhou e o placar do jogo. Para começar um novo jogo, basta clicar com o botão esquerdo em qualquer lugar da tela.

