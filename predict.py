from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os

model = load_model('fishes_46.h5')

diseases_spanish=["atacado_tumor_y_deformidad",  "girodactilo","hidropecia_y_vegiga_nadatoria","huecos en la cabesa",   "nopez","parasito_en_la_boca", "quemadura_bagre", "branquias","gusano_lernea"   ,"hongos","ich_punto_planco","ojo_picho","podredumbre aleta","sanos"]
diseases = ["tumor_and_deformity_attack", "gyrodactylus", "hydropepsia_and_swim_bladder", "head_holes", "nopez", "parasite_in_mouth", "catfish_burn", "gills", "lernea_worm", "fungi", "ich_white_spot", "blind_eye", "fin_rot", "healthy"]

print(diseases)

# target_size_square is the same value as in train.py
target_size_square=256
def predict_image_class(image_path):
	img = image.load_img(image_path, target_size=(target_size_square, target_size_square))
	img_array = image.img_to_array(img)
	img_array = np.expand_dims(img_array, axis=0)
	img_array /= 255.

	prediction = model.predict(img_array)[0]
	max_val=0
	max_pos=0
	print(prediction)
	for i in range(len(prediction)):
		print("  col ",i,prediction,prediction[i])
		if max_val<prediction[i]:
			max_val=prediction[i]
			max_pos=i
	print(prediction,len(prediction),"max value",max_val)
	return diseases[max_pos]
image_path = input("type the name of your image")
predicted_class = predict_image_class(image_path)
print("Predicted class:", predicted_class)
