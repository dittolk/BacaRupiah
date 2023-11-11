############# TRAINING ################

#128 @ 1
python retrain.py --image_dir tf_files/rupiah_photos/ --learning_rate=0.0005 --testing_percentage=15 --validation_percentage=15 --train_batch_size=32 --validation_batch_size=-1 --flip_left_right True --random_scale=30 --random_brightness=30 --eval_step_interval=100 --how_many_training_steps=500 --architecture mobilenet_1.0_128 --output_graph=10_128_graph.pb --output_labels=10_128_labels.txt

#224 @ 1
python retrain.py --image_dir tf_files/rupiah_photos/ --learning_rate=0.0005 --testing_percentage=15 --validation_percentage=15 --train_batch_size=32 --validation_batch_size=-1 --flip_left_right True --random_scale=30 --random_brightness=30 --eval_step_interval=100 --how_many_training_steps=1000 --architecture mobilenet_1.0_224 --output_graph=10_224_graph.pb --output_labels=10_224_labels.txt


############## TESTING #################
#128 @ 
python label_image.py --graph=10_128_graph.pb --labels=10_128_labels.txt --input_layer=input --output_layer=final_result --input_height=128 --input_width=128 --input_mean=128 --input_std=128 --image=2rb.jpg

#224 @ 1
python label_image.py --graph=10_224_graph.pb --labels=10_224_labels.txt --input_layer=input --output_layer=final_result --input_height=224 --input_width=224 --input_mean=224 --input_std=224 --image=DataTest/2rb.jpg

python label_image.py --graph=new_graph3.pb --labels=new_labels3.txt --input_layer=input --output_layer=final_result --input_height=224 --input_width=224 --input_mean=224 --input_std=224 --image=DataTest/2rb.jpg

python label_image.py --graph=model2.pb --labels=labels2.txt --input_layer=input --output_layer=final_result --input_height=224 --input_width=224 --input_mean=224 --input_std=224 --image=DataTest/2rb.jpg



############## SKENARIO #################
@0.25 | 128
@0.5  | 128
@0.75 | 128
@1    | 128

@0.25 | 160
@0.5  | 160
@0.75 | 160
@1    | 160

@0.25 | 192
@0.5  | 192
@0.75 | 192
@1    | 192

@0.25 | 224
@0.5  | 224
@0.75 | 224
@1    | 224


###OPTIMIZE###
python -m tensorflow.python.tools.optimize_for_inference --input=10_224_graph.pb --output=optimized_graph.pb --input_names="input" --output_names="final_result"

##TENSORBOARD##
tensorboard --logdir /tmp/retrain_logs