import numpy as np
import pandas as pd
import sys
import pickle
import os
import chebi

class Clus:

    def __init__(self, clus_folder = 'clus', csv_foler = 'dados/csv', obj_folder='dados/objs', base = 'chebi'):
        self.clus_folder = clus_folder
        self.csv_folder = csv_foler
        self.obj_folder = obj_folder
        self.base = base

        self.hierarquia = chebi.Chebi()
        self.hierarquia = self.hierarquia.load(f'{self.obj_folder}/{base}.obj')

    def cria_header(self):
        
        f = open(f'{self.clus_folder}/{self.base}.arff')
        texto = f.read()
        f.close()
        
        texto = texto.splitlines()

        for i,linha in enumerate(texto):
            linha+='\n'
            if '@data' in linha:
                break

        header_text = ''.join(texto[:i+1]).splitlines()[0].replace('@data', '\n@data\n')
        header_text = header_text.replace('@','\n@')

        f = open(f'{self.clus_folder}/header.txt', 'w')
        f.write(header_text)
        f.close()

    def cria_sfile(self,metodo):

        sfile = f"""
        [Data]
        File = {self.base}.arff
        PruneSet = {self.base}.arff
        TestSet = {metodo}.arff

        [Hierarchical]
        Type = DAG
        WType = ExpAvgParentWeight
        HSeparator = /

        [Tree]
        FTest = [0.001,0.01,0.1,1.0]

        [Model]
        MinimalWeight = 1.0

        [Output]
        WritePredictions = {{Test}}

        """

        file2write=open(f'{self.clus_folder}/{metodo}.s','w')
        file2write.write(sfile)
        file2write.close()

    def cria_arff_alvo(self,metodo):

        f = open(f'{self.clus_folder}/header.txt')
        texto = f.read()
        f.close()

        propriedades = pd.read_csv(f'{self.csv_folder}/{metodo}_propriedades.csv')

        for _,linha in propriedades.iterrows():
            for atributo in linha:
                texto += f'{atributo}, '
            texto += 'CHEBI_24431\n' #dummy target

        file2write=open(f'{self.clus_folder}/{metodo}.arff','w')
        file2write.write(texto)
        file2write.close()
    
    def pred(self,metodo):
        print(f'treinano o classificador hierarquico e realizando as predições para {metodo}')
        dir = os.getcwd()
        alvo = f'{metodo}.s'

        os.chdir(f'{self.clus_folder}')
        sys.stdout = open(os.devnull, 'w')
        os.system(f"java -jar -Xmx16g Clus.jar -xval {alvo}  >nul")
        sys.stdout = sys.__stdout__
        os.chdir(f"{dir}")
        print(f'predições de {metodo} realizadas')
  
    def cria_hierarquia(self,metodo):
        
        saida = f"{self.clus_folder}/{metodo}.test.pred.arff"

        f = open(saida)
        texto = f.read().split('\n\n')
        f.close()

        _,labels,elements_preds = texto

        labels = labels.split('\n')
        labels = labels[1:-1]
        numero_de_classes = len(labels) //2
        labels = labels[numero_de_classes:] 
        labels = [x.split('-')[2].split(' ')[0] for x in labels]

        elements_preds = elements_preds.split('\n')
        elements_preds = elements_preds[1:-1] #o primeiro e ultimo  são a classe prevista e a suposta verdadeira

        threshold = 0.8

        hierarquias_previstas = []
        lb = np.array(labels)
        for elemento in elements_preds:
            bins = elemento.split(',')[1:-1]
            probs = bins[numero_de_classes:]
            preds = bins[:numero_de_classes]
            
            mini = chebi.Chebi()
            
            probs = np.array([float(x) for x in probs])
            labels_do_elemento = lb[ np.where(probs >= threshold)[0]]
            # for a,b in zip(probs,preds): print(a,b)
            for node in labels_do_elemento:
                filhos = self.hierarquia.filhos[node]
                for filho in filhos:
                    if filho in labels_do_elemento:
                        mini.add_edge(node,filho)
            hierarquias_previstas.append(mini)
       
        file = open(f'{metodo}.obj','wb')
        file.write(pickle.dumps(hierarquias_previstas))
        file.close()

        return hierarquias_previstas

    def limpa_clus_folder(self,metodo):
        arquivos = ["hierarchy.txt", f"{metodo}.arff", f"{metodo}.s", f"{metodo}.xval", f"{metodo}.model", f"{metodo}.model", f"{metodo}.out", f"{metodo}.test.pred.arff"]
        for arquivo in arquivos:
            if os.path.exists(f"clus/{arquivo}"):
                os.remove(f"clus/{arquivo}")