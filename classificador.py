# código de knn, pra classificar gatos e cachorros! [emoji feliz]

#as milhares de bibliotecas usadas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def image_to_feature_vector(image, size=(32, 32)):
	# ajusta a imagem para um tamanho fixo
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extraí um histograma da imagem
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	# "normaliza" o histograma de acordo com a versão do opencv etc
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	else:
		cv2.normalize(hist, hist)

	# retorna o histograma
	return hist.flatten()

# construindo os argumentparsers
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# pega as imagens
print("[INFO] verificando imagens...")
imagePaths = list(paths.list_images(args["dataset"]))

# cria matrizes para as imagens "não classificadas" (KKKK), características e labels 
rawImages = []
features = []
labels = []

# loop das imagens 
for (i, imagePath) in enumerate(imagePaths):
	# carrega a imagem e extrai a classe label 
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

        # extraí o "dado não classificado" (acho que é esse seria o nome em 		pt-br de "Raw pixel") e depois o histograma das cores, pra ver qual 		é a distribuição das cores na imagem
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)

	# atualiza as matrizes com os novos dados yay
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)

	# informa sobre o processo a cada mil imagens
	# acho que isso não é tão necessário mas tinha no tutorial e eu quis manter
	if i > 0 and i % 1000 == 0:
		print("[INFO] processadas {}/{}".format(i, len(imagePaths)))

# informa sobre a memória consumida
# minha opinião é a mesma daquele último if das mil imagens kkk
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] matriz de dados não classificados": {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] matriz das características: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))

(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)

# agora sim!! treina o KNN com os "dados não classificados"
print("avaliando precisão dos dados...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("precisão dos dados: {:.2f}%".format(acc * 100))

print("avaliando precisão...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("precisão: {:.2f}%".format(acc * 100))
