#@title imports
from tqdm.auto import tqdm
from collections import defaultdict
from collections import Counter
import re

import pickle
from os.path import exists
import wget

# !pip install kora -q
# import kora.install.rdkit

# import sys
# sys.path.append('/usr/local/lib/python3.7/site-packages/')
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

#https://future-chem.com/rdkit-sa-score/#toc5
import sys
# notebooks/
sys.path.append('/sascorer')

try:
    from sascorer import sascorer
except:
    import sascorer

import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import pylab
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import numpy as np
from collections import defaultdict

#@title `Chebi(file_path):` classe usada para manusear a base de dados(file_path) como um grafo
class Chebi:
  def __init__(self,chebi = ''):

    #@markdown ## **Atributos:**

    #@markdown `tags:` versão mais mastigada do dataset, separado por tags
    self.tags = self.chebi_tags(chebi)
    #@markdown `smiles_tags:` tags dos smiles válidos (inicialmente todos os smilmes, mas posteriormente apenas os que atendem aos criterios)
    self.smiles_tags = []
    #@markdown `smiles:` lista com os smiles
    self.smiles = []
    #@markdown `nodes:` conjunto de vertices do grafo
    self.nodes = set()
    #@markdown `pais:` dicionario, chave é um vertice, valor é uma lista com os pais desse vertice
    self.pais = defaultdict(list)
    #@markdown `filhos:` dicionario, chave é um vertice, valor é uma lista com os filhos desse vertice
    self.filhos = defaultdict(list)
    #@markdown `distances:` não usado, mas comummente usado para grafos e muito esperto, logo não removido
    self.distances = {}
    #@markdown `raizes:` conjunto de vertices raizes da base chebi    ['strcutture', 'particle', 'role'] (vi no site pra saber quais os ids) []
    self.raizes = ['CHEBI_24431','CHEBI_36342','CHEBI_50906']
    #@markdown `folhas:` conjunto de folhas do grafo
    self.folhas = set()
    #@markdown `excluded_nodes:` conjunto de vertices podados
    self.excluded_nodes = set()
    #@markdown `irregulares:` nós que não atendem aos criterios e que são podados 
    self.irregulares = ['CHEBI_33286','CHEBI_64857','CHEBI_37958','CHEBI_64047','CHEBI_48318','CHEBI_47867','CHEBI_25944','CHEBI_136643','CHEBI_50406','CHEBI_33893','CHEBI_78433','CHEBI_46787','CHEBI_35204','CHEBI_36342']
    #@markdown `count:` usado para contar quantas instacias cada rotulo possui
    self.count = Counter()
    #@markdown `count_folhas:`
    self.count_folhas = Counter()
    #@markdown `atualizado:` usado pelos algoritmos recursivos para verificar  se um nó já foi visitado para evitar recontagens
    self.atualizado = defaultdict(lambda:False)
    #@markdown `pertencente:` tentando usar programação dinamica nas funções has ancestral
    self.pertencente = defaultdict(lambda:False) # -1 não testado, 0 testado e ta fora, 1 testado e ta dentro 
    #@markdown `profundidade:` variavel memoria para programação dinamica, usada para otimizar a função
    self.profundidade_no = defaultdict(lambda:0)
    #@markdown `classes_expressao:` expressão regular usada para extrair as classes de uma instância
    self.classes_expressao = re.compile(r'<.*?:([a-zA-z]*) .*?(CHEBI_[0-9]*)')
    #@markdown `atributos_expressao:` expressão regular usada para extrair os atributos de uma instância(mais pra pegar os smiles mesmo)
    self.atributos_expressao = re.compile(r'<\w*?:(\w*?) [\w:=#"\/\._]*?">(.*?)<')

    self.texto = ''

    self.estatistica = {
        "profundidade":0,
        "quantidade de nós":0,
        "quantidade de folhas":0,
        "maior numero de filhos":0,
        "pai com maior numero de filhos": "chebi_algum",
        "maior numero de pais": 0,
        "filho com maior numero de pais": "chebi_loko",
        "média de filhos" : 0,
        "média de pais" : 0
    }
    # self.estatistica2 = self.estatistica.copy()

    self.count_relacoes = 0
  
  #@markdown ## **Funções**

  
  #@markdown `chebi_tags(chebi):` lê e a base de dados(`chebi`) e a divide em tags, onde cada tag representa uma intancia, e suas subtags representam seus atributos 
  def chebi_tags(self,file_path):
    if file_path == '':
      return ['']

    if exists(path_to_file):
      try:
        with open(file_path, 'r') as file:
          data = file.read().replace('\n', '')
      except:
        print('não foi possivel abrir o aqrquivo ',file_path)
    else:
      try:
        url = "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl"
        filename = wget.download(url)
        with open(file_name, 'r') as file:
          data = file.read().replace('\n', '')
      except
        print('erro ao baixar')
        return ['']

    data = re.sub(r'\s+',' ', data) #substituindo espaços duplos, quebras de linha e etc por um espaço único
    data = re.sub(r'<owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/BFO_0000051"/> <owl:someValuesFrom','<owl:hasPart', data)  #a tag em questão não é facilmente interpretavel e não estava sendo pega pelo regex, então foi alterada para melhor interpretação
    data = re.sub(r'<owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/RO_0000087"/> <owl:someValuesFrom','<owl:hasRole', data)   #a tag também apresentava o mesmo problema

# data = re.sub(r'owl:someValuesFrom','rdfs:subClassOf',data)

    express = re.compile(r'<owl:Class.*?</owl:Class>')

    return express.findall(data)

  
  #@markdown `add_node(vertice):` função para adcionar um novo `vertice` na hierarquia
  def add_node(self, value):
    self.nodes.add(value)
    # self.count[value] = 0 #de dict para counter isso n é mais preciso
    self.atualizado[value] = False
    # self.count.update({value,0})

  
  #@markdown `add_edge(pai,filho):` função para adcionar relação entre dois vertices
  def add_edge(self, from_node, to_node, distance=1):
    '''
      Adciona um vertice 
    '''
    if from_node == to_node:
      return False

    self.add_node(from_node)
    self.add_node(to_node)
    if to_node not in self.filhos[from_node]:
      self.filhos[from_node].append(to_node)
      self.pais[to_node].append(from_node)
    self.distances[(from_node, to_node)] = distance
    self.distances[( to_node,from_node)] = distance

  
  #@markdown `zera_atualizado():` desmarca todos os nós, para que uma função recursiva possa ir remarcando-os de acordo com seu algoritmo
  def zera_atualizado(self):
    self.atualizado = defaultdict(lambda:False)
  
  #@markdown `zera_profundidade`
  def zera_profudidade(self):
    self.profundidade_no = defaultdict(lambda:0)

  
  #@markdown `create_hierarchy():` cria a hierarquia com rotulos(instancias sem smiles) 
  def create_hierarchy(self):
    '''
    Cria a hierarquia com rotulos(instancias sem smiles) 
    primeiro verifica se a tag possui smiles
    se não possui é considerada como rótulo e adcionado à hierarquia
    caso possua é considerado como instância

    Existe um problema pois alguns rotulos possuem smiles, estes geream subgrafos, o que é corrigido com correct_hierarchy()  
    ainda não foi encontrado em que momento o subgrafo é gerado, um dia, prevenir pode ser melhor que remediar 
    '''
    pbar = tqdm(self.tags)
    pbar.set_description("Criando hierarquia")
    for tag in pbar:

      atributos = self.atributos_expressao.findall(tag)
      marcadorDeAtributos=[]
      for relacao,atributo in atributos:
        marcadorDeAtributos.append(relacao)

      if 'smiles' not in marcadorDeAtributos:

        classes = self.classes_expressao.findall(tag)
        id = classes[0][1]

        for relacao,classe in classes[1:]:
          if relacao == 'subClassOf' or relacao =='hasRole': #hasRole não é necessario mas vai q né, depois retirar 
            self.add_edge(classe,id,1)
      else:
        self.smiles_tags.append(tag)

    self.correct_hierarchy()

  
  #@markdown `correct_hierarchy():` conecta subgrafos (criados erroneamente) ao subgrafo devido
  def correct_hierarchy(self):
    pbar = tqdm(self.get_nos_sem_pais())
    pbar.set_description("Corrigindo hierarquia")

    for sem_pai in pbar:      
      for tag in self.tags:
        if sem_pai in tag:
          classes = self.classes_expressao.findall(tag)
          if sem_pai == classes[0][1]:
            for filho in self.filhos[sem_pai]:
              for prop,classe in classes[1:]:                
                self.add_edge(classe,sem_pai)
    self.remove_cicles()
              
  
  #@markdown `remove_cicles:` poda vertices que criam ciclos curtos, onde são pais do prorpio pai 
  def remove_cicles(self):
    siameses = defaultdict(set) #quando um nó é seu pŕoprio avô, outro nó tambem é seu proprio avo para fechar o ciclo, isso acontece em pares(ou trios, ou grupos) esses grupos são siameses entre si, pois um "é" outro
    proprios_avos = [] 
    for node in self.nodes:
        pais = self.pais[node]
        for pai in pais:
            proprio_vo = self.pais[pai]
            if node in proprio_vo:
              proprios_avos.append(node)
              siameses[node].add(pai)
              siameses[pai].add(node)
              siameses[node].add(node)
    for siames in siameses.values():
        pai_uniao = {x for y in siames for x in self.pais[y] if x not in siames}
        filho_uniao = {x for y in siames for x in self.filhos[y] if x not in siames}
        
        for node in siames:
            self.pais[node] = pai_uniao
            self.filhos[node] = filho_uniao

  
  #@markdown `get_nos_sem_pais():`
  def get_nos_sem_pais(self):
    sem_pais = []
    for node in self.nodes:
      if len(self.pais[node]) is 0 and node not in self.raizes:
        sem_pais.append(node)
    return sem_pais

  
  #@markdown `shrink_hierarchy():`
  def shrink_hierarchy(self):

    for no in self.nodes:
      if len(self.filhos[no]) is 1 and len(self.filhos[no]) is 1:
        pass

  
  #@markdown `has_ancestral(rorulos,alvo):` função que determina se uma lista de rotulos(`rotulos`) possuem ancrestralidade de um determinado `alvo` na hierarquia
  def has_ancestral(self,rotulos,alvo):
    if type(rotulos) != type([]): #funciona apenas com vetores, se vier um valor, faz ser um vetor
      rotulos = [rotulos]
    
    #blockin novo
    lista =[]    
    for rotulo in rotulos:

      if self.pertencente[(rotulo,alvo)] == -1:
        self.pertencente[(rotulo,alvo)] = self.has_ancestral2(rotulo,alvo)

      lista.append(self.pertencente[(rotulo,alvo)])
      
    
    return lista

    return [self.has_ancestral2(x,alvo) for x in rotulos]

  
  #@markdown `has_ancestral2(rotulo,alvo):` função que determina se um `rotulo` possui ancrestralidade de um determinado `alvo` na hierarquia
  def has_ancestral2(self,rotulo,alvo):
    if rotulo not in self.nodes:
      return False
    pais = self.pais[rotulo]
    if len(pais) == 0:
      return False
    if alvo in pais:
      return True

    #blockin novo
    if self.pertencente[(rotulo,alvo)] == -1:
      self.pertencente[(rotulo,alvo)] = True in self.has_ancestral(pais,alvo)

    return self.pertencente[(rotulo,alvo)]
    
    
    return True in self.has_ancestral(pais,alvo)

  
  #@markdown `marca_parente():`
  def marca_pertencente(self,alvo,rotulo=None):
      if rotulo == None:
          rotulo = alvo
      if self.pertencente[(rotulo,alvo)]:
        return 1

      self.pertencente[(rotulo,alvo)] = True
      for filho in self.filhos[rotulo]:
          self.marca_pertencente(alvo,filho)

  
  #@markdown as funções `has_ancestral(rorulos,alvo)` e `has_ancestral2(rotulo,alvo)` dependem uma da outra para funcionar
  def zera_pertencente(self):
    self.pertencente = defaultdict(lambda:False)

  
  #@markdown `att_count_raiz():` função para contar quantas instâncias cada rotulo possui
  def att_count_raiz(self):
    excecoes = []
    #reseta a contagem
    for id in self.count:
      self.count[id] = 0
      self.atualizado[id] = False

    #conta as intanscias das "folhas"
    pbar = tqdm(self.smiles_tags)
    pbar.set_description("atualizando marcadores")
    for tag in pbar:

      classes = self.classes_expressao.findall(tag)
      
      for relacao,classe in classes[1:]:
        if relacao == 'subClassOf' or relacao == 'hasRole':
          if classe in self.nodes:
            self.count_folhas.update(classe.split())
            self.count.update(classe.split()) # o split serve pra contar a palavra e não as letras
          else:
            excecoes.append(classe)

    #conta recursivamente o numero de filhos(acaba repetindo, é mais pra ter uma ideia de grandeza)    
    soma = 0
    for id in self.raizes:    
      if id in self.nodes:
        if not self.atualizado[id]:
          for filho in self.filhos[id]:
            soma += self.att_count(filho) 
          self.count.update({id:soma})
          self.atualizado[id] = True
    # print('hmm', excecoes)

  
  #@markdown `att_count_raiz(vertice):` função recursiva para contar quantas instancias cada rotulo possui
  def att_count(self,id):
    soma = 0

    if not self.atualizado[id]:

      for filho in self.filhos[id]:
        soma += self.att_count(filho) 
      self.count.update({id:soma}) #self.count[id] += soma
      self.atualizado[id] = True

    return self.count[id]

  
  #@markdown `poda(vertice):` remove um `vertice` e seus filhos da hierarquia recursivamente
  def poda(self,id):
    if id in self.excluded_nodes:
      return

    
    a_podar = []
    for filho in self.filhos[id]:
      a_podar.append(filho)
    for no in a_podar:
      self.poda(no)

    for pai in self.pais[id]:
      try:
        self.filhos[pai].remove(id)
      except:
        pass

    dicionarios = [self.atualizado,self.count,self.filhos,self.pais]
    for dicionario in dicionarios:
      try:
        del(dicionario[id])
      except:
        pass
    
    try:
      self.excluded_nodes.add(id)
      self.nodes.remove(id)
    except:
      pass
    
  
  #@markdown `poda_irregulares(criterio)` Poda os rótulos que possuem menos smilesque que um determinado `criterio`, ou os que foram considerados fora do escopo (`irregulares`) 
  def poda_irregulares(self,criterio=0, irregulares = []):
    
    for irregular in irregulares:
      self.irregulares.append(irregular)

    numeroDePodados = 0 
    while numeroDePodados != len(self.irregulares):
      numeroDePodados = len(self.irregulares)

      self.atualiza_smiles()
      self.att_count_raiz()

      for rotulo in self.count:
        if self.count[rotulo] < criterio:
          self.irregulares.append(rotulo)

      for rotulo in self.irregulares:
        self.poda(rotulo)

      contador = Counter(self.count.values())
      # print(contador)
      # print(self.count)

  
  #@markdown `atualiza_smiles():` seleciona os smiles que são decendentes de role
  def atualiza_smiles(self):

    self.zera_pertencente()
    self.marca_pertencente(self.raizes[2])

    new_smiles_tags = []
    pbar = tqdm(self.smiles_tags)
    pbar.set_description("Atualizando smiles")

    for tag in pbar:

      classes = self.classes_expressao.findall(tag)
      classesId = []
      for relacao,classe in classes[1:]:
        classesId.append(classe)

      role_classes = self.has_ancestral(classesId,self.raizes[2])
      if True in role_classes:
        new_smiles_tags.append(tag)

    self.smiles_tags = new_smiles_tags

  
  #@markdown `set_smiles()`: faz uma lista com os smiles validos usando `smiles_tags` 
  def set_smiles(self):  
    smiles = []
    for tag in self.smiles_tags:
      atributos = self.atributos_expressao.findall(tag)
      for relacao,atributo in atributos:
        if relacao == 'smiles':
          smiles.append(atributo)

    self.smiles = smiles
    return self.smiles

  
  #@markdown `save_smiles(dir):` salva os smiles válidos no diretorio `dir`
  def save_smiles(self,dir):
    
    smiles_txt = ''

    for smile in self.smiles:
      smiles_txt+=smile+'\n'

    file2write=open(dir,'w')
    file2write.write(smiles_txt)
    file2write.close()

    return smiles_txt

  
  #@markdown `save_arff(dir):` cria o arquivo arff e o salva em um diretrio(`dir`)
  def save_arff(self,dir):
    texto = '''@relation simple

    @attribute peso numeric
    @attribute area numeric
    @attribute solubilidade numeric
    @attribute druglikness numeric
    @attribute SAscore numeric
    @attribute HBA numeric
    @attribute HBD numeric
    @attribute class hierarchical '''

    self.zera_atualizado()

    for raiz in self.raizes:
      if raiz in self.nodes:
        texto += self.texto_hierarquia(raiz)

    texto += self.texto_data()

    self.texto = texto

    file2write=open(dir,'w')
    file2write.write(self.texto)
    file2write.close()
    return texto

  
  #@markdown `texto_hierarquia(id):` cria um texto com a hierarquia apartir de um vertice(`id`), representando no padrão dos arquivos .arff
  def texto_hierarquia(self,id):
    texto = ''
    if self.atualizado[id]:
      return texto

    for filho in self.filhos[id]:
      texto+= id+ '/' + filho+','
      self.count_relacoes+=1
      texto += self.texto_hierarquia(filho)
    self.atualizado[id] = True
    return texto

  
  #@markdown `texto_data():` cria um texto com as propriedades das moléculas e seus rótulos
  def texto_data(self):

    self.zera_pertencente()
    for raiz in self.raizes:
      if raiz in self.nodes:
        self.marca_pertencente(raiz)
    
    texto_data = '''
      @data

    '''

    descritores = [
      Descriptors.MolWt,
      Descriptors.TPSA,
      Descriptors.MolLogP,
      Descriptors.qed,
      sascorer.calculateScore,
      Chem.rdMolDescriptors.CalcNumLipinskiHBA,
      Chem.rdMolDescriptors.CalcNumLipinskiHBD
    ]
    
    pbar = tqdm(self.smiles_tags)
    pbar.set_description("Processando smiles")
    for tag in pbar:

      #bloco para guardar as classes da instancia
      classes = self.classes_expressao.findall(tag)
      classesId = []
      for relacao,classe in classes[1:]:
        classesId.append(classe)

      #bloco pra pegar o SMILES da instancia
      atributos = self.atributos_expressao.findall(tag)
      for relacao,atributo in atributos[1:]:
        if relacao=='smiles':
          smiles = atributo
          break

      try:

        #bloco para selecionar as classes que ainda fazem parte da hierarquia
        role_classes = self.has_ancestral(classesId,self.raizes[2])
        chemichal_classes = self.has_ancestral(classesId,self.raizes[0])
        texto_classes = ''
        tem_classe = False
        for classe,role,chemichal in zip(classesId,role_classes,chemichal_classes):
          if role or chemichal:
            texto_classes += classe+'@'
            tem_classe = True
          if not tem_classe:
            continue

        #bloco pra calcular os descritores dos smiles
        texto_numerico = ''
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

        for descritor in descritores:
          texto_numerico += str( descritor(mol) )+', '

        texto_data += texto_numerico + texto_classes[:-1] +'\n'
      except:
        continue

    return texto_data

  
  #@markdown `faz_estatisticas(id):` calcula as estatisticas de um grafo apartir de sua raiz(`id`)
  def faz_estatisticas(self,id):    
    if id not in self.nodes:
      return ''

    total ={
        'folhas':0,
        'pais':0,
        'filhos':0
    }
    maior ={
        'pais':0,
        'idp':'',

        'filhos':0,
        'idf':''
    }

    self.zera_profudidade()
    self.estatistica["profundidade"] = self.profundidade(id)
    self.zera_atualizado()
    self.estatistica["quantidade de nós"] = self.conta_filhos(id)
    self.zera_atualizado()
    self.mais_parentes(id,maior,total)
    
    self.estatistica['maior numero de filhos'] = maior['filhos']
    self.estatistica['pai com maior numero de filhos'] = maior['idf']
    self.estatistica['maior numero de pais'] = maior['pais']
    self.estatistica['filho com maior numero de pais'] = maior['idp']
    self.estatistica['quantidade de folhas'] = total['folhas']
    self.estatistica['média de filhos'] = total['filhos'] / self.estatistica["quantidade de nós"]

    
    return self.estatistica.copy()

  
  #@markdown `profundidade(id):` calcula a profundidade da Dag apartir de um nó(`id`)
  def profundidade(self,id):
    if self.profundidade_no[id] > 0:
      return self.profundidade_no[id]

    if len(self.filhos[id]) == 0:
      self.profundidade_no[id] = 1
      return 1

    profundidades = []
    for filho in self.filhos[id]:
      profundidades.append(self.profundidade(filho))

    self.profundidade_no[id] = 1 + np.array(profundidades).max()
    return self.profundidade_no[id]

  
  #@markdown `sobe(): função utilizada para calcular a altura dos rotulos na DAG`
  def sobe(self,node,nova_altura,fila):
    if nova_altura < self.profundidade_no[node]:
      return
    
    self.profundidade_no[node] = nova_altura

    for pai in self.pais[node]:

      fila.append(
        (pai,nova_altura+1)
        )
      
  
  #@markdown `calcula_profundidades(): calcula a profundidade de todos os rotulos da forma mais interativa que conseguir pensar, pois recursivo estoura a pilha `
  def calcula_profundidades(self):
      fila = []
      self.zera_profudidade()
      for node in self.nodes:
        if len(self.filhos[node]) == 0:
          fila.append((node,1))

      while fila:
        node, altura = fila.pop(0)
        self.sobe(node,altura,fila)

  
  #@markdown `conta_filhos(vertice):` função que conta quantos filhos cada vertice tem recursivamente, basicamente quantos nós tem no sub-grafo
  def conta_filhos(self,id):
    if self.atualizado[id]:
      return 0
    self.atualizado[id] = True
    
    filhos = 1 #se inclui na contagem pois o barato é loko
    for filho in self.filhos[id]:
      filhos += self.conta_filhos(filho)
    return filhos
    
  
  #@markdown `mais_parentes():` função forçadinha pra fazer as estatisticas, poderia ser mais modular mas maior preguiça
  def mais_parentes(self,
  id,maior,total):
    if self.atualizado[id]:
      return 
    self.atualizado[id] = True

    numFilhos = len(self.filhos[id])
    numPais = len(self.pais[id])
    total['filhos'] += numFilhos
    total['pais'] += numPais
    if numFilhos == 0:
      total['folhas'] += 1
      self.folhas.add(id)

    for filho in self.filhos[id]:
      self.mais_parentes(filho,maior,total)

    if numFilhos > maior['filhos']:
      maior['filhos'] = numFilhos
      maior['idf'] = id 
    if numPais > maior['pais']:
      maior['pais'] = numPais
      maior['idp'] = id

    def printa(self,node,nivel=0,chebi=self):
        if chebi.atualizado[node]:
            return
        chebi.atualizado[node] = True
        print(f"{'| '*nivel}├{node}")
        for filho in chebi.filhos[node]:
            printa(filho,nivel=nivel+1,chebi=chebi)
    
    def save(self,file_name):
        """save class as file_name"""
        file = open(file_name,'w')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

    def load(self,file_name):
        """try load file_name"""
        try:
            file = open(file_name,'r')
            dataPickle = file.read()
            file.close()
            self.__dict__ = cPickle.loads(dataPickle)
        except:
            print(f'{file_name} não encontrado')    