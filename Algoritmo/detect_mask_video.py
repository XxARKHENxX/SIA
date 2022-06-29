# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# pegue as dimensões do quadro e construa um blob
	# a partir dele
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# passe o blob pela rede e obtenha as detecções de rosto
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# inicialize uma lista de rostos, suas localizações correspondentes,
	# a lista de previsões de nossa rede de máscaras faciais
	faces = []
	locs = []
	preds = []

	# loop sobre as detecções
	for i in range(0, detections.shape[2]):
		# extrair a confiança (ou seja, probabilidade) associada a
		# a detecção
		confidence = detections[0, 0, i, 2]

		# filtre detecções fracas, garantindo que a confiança seja
		# maior do que a confiança mínima
		if confidence > 0.5:
			# calcule as coordenadas (x, y) da caixa delimitadora para
			# o objeto
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# garantir que as caixas delimitadoras estejam dentro das dimensões de
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extraia o ROI da face, converta-o de BGR para canal RGB
			# pedido, redimensione-o para 224x224 e pré-processe-o
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# adicione a face e as caixas delimitadoras aos seus respectivos
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# só faça previsões se pelo menos um rosto for detectado
	if len(faces) > 0:
		# para inferência mais rápida, faremos previsões em lote em *todos*
		# rostos ao mesmo tempo, em vez de previsões um a um
		# acima `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# retornar uma tupla de 2 dos locais das faces e seus correspondentes
	# Localizações
	return (locs, preds)

# carregue nosso modelo de detector de rosto serializado do disco
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# carregue o modelo do detector de máscara facial do disco
maskNet = load_model("mask_detector.model")

# inicializar o fluxo de vídeo
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop sobre os quadros do fluxo de vídeo
while True:
	# pegue o quadro do fluxo de vídeo encadeado e redimensione-o
	# ter uma largura máxima de 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detectar rostos no quadro e determinar se eles estão usando um
	# máscara facial ou não
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop sobre os locais de rosto detectados e seus correspondentes
	# locations
	for (box, pred) in zip(locs, preds):
		# descompacte a caixa delimitadora e as previsões
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine o rótulo e a cor da classe que usaremos para desenhar
		# a caixa delimitadora e o texto
		count = 1
		label = "Mask teste" if mask > withoutMask else "No Mask teste" and cv2.imwrite("C:\SIA\Algoritmo\print/teste%d.jpg" % count, frame)
		count += 1
		color = (0, 255, 0) if label == "Mask teste" else (0, 0, 255)

		# inclua a probabilidade no rótulo
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# exibir o rótulo e o retângulo da caixa delimitadora na saída
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# mostrar o quadro de saída
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# se a tecla `q` foi pressionada, saia do loop
	if key == ord("q"):
		break

# fazer um pouco de limpeza
cv2.destroyAllWindows()
vs.stop()