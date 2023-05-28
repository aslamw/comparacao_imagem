"""
 detecção e descrição de características utilizando o algoritmo ORB (Oriented FAST and Rotated BRIEF) do OpenCV. 
 As características das imagens de referência e consulta são detectadas e descritas usando orb.detectAndCompute().
 Em seguida, utilo o objeto cv2.BFMatcher para realizar a correspondência das características entre as imagens.
"""

import cv2

# Carrega a imagem de referência e a imagem de consulta
reference_image = cv2.imread('cat.jpg')
query_image = cv2.imread('base_cat.jpg')

# Converte as imagens para escala de cinza
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

# Cria o detector de características ORB
orb = cv2.ORB_create()

# Detecta e descreve as características da imagem de referência
kp_reference, des_reference = orb.detectAndCompute(reference_gray, None)

# Detecta e descreve as características da imagem de consulta
kp_query, des_query = orb.detectAndCompute(query_gray, None)

# Cria o objeto de correspondência de características
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Realiza a correspondência das características entre as imagens
matches = bf.match(des_reference, des_query)

# Ordena as correspondências com base na distância
matches = sorted(matches, key=lambda x: x.distance)

# Desenha as correspondências mais próximas em uma nova imagem
matching_result = cv2.drawMatches(reference_image, kp_reference, query_image, kp_query, matches[:10], None, flags=2)

# Exibe o resultado das correspondências
cv2.imshow('Matching Result', matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
