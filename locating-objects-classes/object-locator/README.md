This implementation is based of the work done by Ribera.
Please see the following link for any code and copyright information: 
https://github.com/javiribera/locating-objects-without-bboxes


Differences in files:
	New Files:
		All files in the datapreprocessing page are my own. 
		These were made from scratch to process images from
		the MPII dataset.
		Directory: locating-objects-classes/data_preprocessing/
		Code Files: mat2csv.m, clean_csv.py
	Heavily Modified files:
		The main training, model and testing files were heavily
		modified to support training and testing on multiple files.
		Directories: locating-objects-classes/object-locator/models;
			     locating-objects-classes/object-locator
		Code Files: unet_model.py, train.py, locate.py
	Slightily Modified Files:
		Some other files have slight modifications just to support higher 
		dimension variables.
	Reused code:
		The rest of the utility code, such as the painter and thresholder was reused.


How to Use Code:
	To download the pretrained model please use this link:
		https://drive.google.com/file/d/1gp_zC8Mps4_-6xxBJG8pmuaSuNqWXTya/view?usp=sharing
		if this link ever goes off line please let me know in the issues section of this 			github repo

	To train code please update lines 73 and 83 of train.py with the correct paths to the 		images and save locations for the model checkpoints. Then simply run the python script.

	To test the implementation please update lines 71-73 of test.py with the correct paths
	to the images and save locations for the model checkpoints. Then simply run the python 		script.

	Goodluck!

	
