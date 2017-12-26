# load the val annotations file

import os

def get_annotations_map():
	valAnnotationsPath = '/home/student-9/PycharmProjects/Axon-Course/data_dir/miny_set/tiny-imagenet-200/val/val_annotations.txt'
	valAnnotationsFile = open(valAnnotationsPath, 'r')
	valAnnotationsContents = valAnnotationsFile.read()
	valAnnotations = {}

	for line in valAnnotationsContents.splitlines():
		pieces = line.strip().split()
		valAnnotations[pieces[0]] = pieces[1]

	return valAnnotations