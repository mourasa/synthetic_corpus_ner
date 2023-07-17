import csv
import re
import os
import sys

os.chdir('/home/getalp/mourasa/pipeline/4_annotation')

clean = open('../results/'+str(sys.argv[1])+'/annotated_texts.txt','w')
clean.close()

# Annotate entities
def annotate_entities(matrix_csv, generated_texts, output_file):
    with open(matrix_csv, 'r', encoding='utf-8') as csv_file:
      reader=csv.DictReader(csv_file)
      people_set=set()
      place_set=set()
      org_set=set()

      for row in reader:
        person=row['PERSON']
        place=row['PLACE']
        organization=row['ORGANIZATION']
        people_set.add(person)
        place_set.add(place)
        org_set.add(organization)

    with open(generated_texts, 'r', encoding='utf-8') as text_file:
      texts=text_file.readlines()

    annotated_texts=[]
    for text in texts:
        for person in people_set:
            pattern = re.compile(fr"\b({re.escape(person)})\b")
            text = re.sub(pattern, r'<PERSON>\g<0></PERSON>', text)
        for place in place_set:
            pattern=re.compile(fr"\b({re.escape(place)})")
            text=re.sub(pattern, r'<PLACE>\g<0></PLACE>', text)
        for org in org_set:
            pattern=re.compile(fr"\b({re.escape(org)})")
            text=re.sub(pattern,r'<ORGANIZATION>\g<0></ORGANIZATION>', text)

        annotated_texts.append(text)

    with open(output_file, 'a') as output:
        output.writelines(annotated_texts)


for k in range(0,int(sys.argv[2])):
  print("Annotation en cours :")
  print(k)
  # Annotation example
  annotate_entities('../results/'+str(sys.argv[1])+'/matrix.csv','../results/'+str(sys.argv[1])+'/generation/generation_prompt_%d.txt' %k, '../results/'+str(sys.argv[1])+'/annotated_texts.txt')




text_file='../results/'+str(sys.argv[1])+'/annotated_texts.txt'
# Read the annotated_texts file
with open(text_file, 'r', encoding='utf-8') as txt_file:
  texts=txt_file.readlines()
# Put the filtered texts in a list
filtered_texts=[]
for text in texts:
  # If at least two tags are in a text, append it to the filtered_texts list
  if ((('<PERSON>' in text) and ('<ORGANIZATION>' in text)) or (('<PERSON>' in text) and ('<PLACE>' in text)) or (('<ORGANIZATION>' in text) and ('<PLACE>' in text))):
    filtered_texts.append(text)
    # Store the filtered texts in a new txt file
    with open('../results/'+str(sys.argv[1])+'/filtred_texts.txt', 'w', encoding='utf-8') as txt_file:
      txt_file.writelines(filtered_texts)





#Ouverture en Lecture du fichier de annotated_texts.txt
annotated_texts = open('../results/'+str(sys.argv[1])+'/filtred_texts.txt','r')

#Création du fichier de destination
output = open('../results/'+str(sys.argv[1])+'/corpus_filtred.txt','w')

marquage = 'O'

#Parcours de chaque ligne
for ligne in annotated_texts:
  #Parcours de chaque mot de la ligne
  ligne = ligne.replace('\''," ' ")
  ligne = ligne.replace("."," .")
  ligne = ligne.replace(","," ,")
  ligne = ligne.replace("’"," ’ ")
  ligne = ligne.replace(":"," :")
  ligne = ligne.replace("\""," \" ")
  ligne = ligne + '\n'
  for mot in ligne.split():
    #Affectation O
    if (marquage == 'O'):
      bloc = mot + " O\n"

    #Vérification
    if (marquage == 'P'):
      if ("</PERSON>" in mot):
        bloc = mot[:-9] + " I-PER\n"
        marquage = 'O'
      else:
        bloc = mot + " I-PER\n"

    elif (marquage == 'L'):
      if ("</PLACE>" in mot):
        bloc = mot[:-8] + " I-LOC\n"
        marquage = 'O'
      else:
        bloc = mot + " I-LOC\n"

    elif (marquage == 'G'):
      if ("</ORGANIZATION>" in mot):
        bloc = mot[:-15] + " I-ORG\n"
        marquage = 'O'
      else:
        bloc = mot + " I-ORG\n"

    if ("<PERSON>" in mot):
      if ("</PERSON>" in mot):
        bloc = mot[8:-9] + " B-PER\n"
        marquage = 'O'
      else:
        bloc = mot[8:] + " B-PER\n"
        marquage = 'P'

    elif ("<PLACE>" in mot):
      if ("</PLACE>" in mot):
        bloc = mot[7:-8] + " B-LOC\n"
        marquage = 'O'
      else:
        bloc = mot[7:] + " B-LOC\n"
        marquage = 'L'

    elif ("<ORGANIZATION>" in mot):
      if ("</ORGANIZATION>" in mot):
        bloc = mot[14:-15] + " B-ORG\n"
        marquage = 'O'
      else:
        bloc = mot[14:] + " B-ORG\n"
        marquage = 'G'


    #Ecriture du mot et de son annotation dans le fichier de destination
    if (mot != "\n"):
      output.write(bloc)
  #Ecriture d'un saut de ligne dans le fichier de destination
  output.write('\n')

annotated_texts.close()
output.close()

annotated = open('../results/'+str(sys.argv[1])+'/annotated_texts.txt','r')
filtred = open('../results/'+str(sys.argv[1])+'/filtred_texts.txt','r')
print("Taille du corpus initial : " + str(len(annotated.readlines())))
print("Taille du corpus filtré : " + str(len(filtred.readlines())))
annotated.close()
filtred.close()


