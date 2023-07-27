import numpy as np
 
# use for index 2 rgb
floorplan_room_map = {
	0: [  0,  0,  0], # background
	1: [192,192,224], # closet
	2: [192,255,255], # bathroom/washroom
	3: [224,255,192], # livingroom/kitchen/diningroom
	4: [255,224,128], # bedroom
	5: [255,160, 96], # hall
	6: [255,224,224], # balcony
	7: [224,224,224], # not used
	8: [224,224,128]  # not used
}

floorplan_furn_map = {
	0: [0, 0, 0], #background
	1: [  12,  34,  56], # cabinet
	2: [78, 90, 102], # table
	3: [114, 126, 138], # bookshelf
	4: [150, 162, 174], # counter
	5: [186, 198, 210], # desk
	6: [222, 234, 246], # shelves
	7: [247, 123, 189], # dresser
	8: [137, 99, 203], # mirror
	9: [59, 171, 93],  # fridge
	10: [125, 231, 127], # television
	11: [189, 17, 157], # box
	12: [77, 181, 219], # whiteboard
	13: [33, 75, 141], # night stand
	14: [211, 45, 71], # bed
	15: [175, 107, 35], # chair
	16: [63, 209, 243], # sofa
	17: [29, 141, 55], # toiled
	18: [95, 201, 111], # sink
    19: [159, 5, 163] # bathtub
}

tf2deep_furn_cmap = {
    # obstacles
    'cabinet': [12/255, 34/255, 56/255, 1], 
    'table': [78/255, 90/255, 102/255, 1], 
    'bookshelf': [114/255, 126/255, 138/255, 1],    
    'counter': [150/255, 162/255, 174/255, 1], 
    'desk': [186/255, 198/255, 210/255, 1], 
    'shelves': [222/255, 234/255, 246/255, 1], 
    'dresser': [247/255, 123/255, 189/255, 1], 
    'mirror': [137/255, 99/255, 203/255, 1], 
    'fridge': [59/255, 171/255, 93/255, 1],
    'television': [125/255, 231/255, 127/255, 1], 
    'box': [189/255, 17/255, 157/255, 1], 
    'whiteboard': [77/255, 181/255, 219/255, 1], 
    'night stand': [33/255, 75/255, 141/255, 1], # removed 'structure', 'furniture', 'prop' because they are too big
    # center_bbox:
    'bed': [211/255, 45/255, 71/255, 1],
    'chair': [175/255, 107/255, 35/255, 1], 
    'sofa': [63/255, 209/255, 243/255, 1], 
    'toilet': [29/255, 141/255, 55/255, 1], 
    'sink': [95/255, 201/255, 111/255, 1], 
    'bathtub': [159/255, 5/255, 163/255, 1]
}

# boundary label
floorplan_boundary_map = {
	0: [  0,  0,  0], # background
	1: [255,60,128],  # opening (door&window)
	2: [255,255,255]  # wall line	
}

# boundary label for presentation
floorplan_boundary_map_figure = {
	0: [255,255,255], # background
	1: [255, 60,128],  # opening (door&window)
	2: [  0,  0,  0]  # wall line	
}

# merge all label into one multi-class label
floorplan_fuse_map = {
	0: [  0,  0,  0], # background
	1: [192,192,224], # closet
	2: [192,255,255], # batchroom/washroom
	3: [224,255,192], # livingroom/kitchen/dining room
	4: [255,224,128], # bedroom
	5: [255,160, 96], # hall
	6: [255,224,224], # balcony
	7: [224,224,224], # not used
	8: [224,224,128], # not used
	9: [255,60,128],  # extra label for opening (door&window)
	10: [255,255,255]  # extra label for wall line
}

# invert the color of wall line and background for presentation
floorplan_fuse_map_figure = {
	0: [255,255,255], # background
	1: [192,192,224], # closet
	2: [192,255,255], # batchroom/washroom
	3: [224,255,192], # livingroom/kitchen/dining room
	4: [255,224,128], # bedroom
	5: [255,160, 96], # hall
	6: [255,224,224], # balcony
	7: [224,224,224], # not used
	8: [224,224,128], # not used
	9: [255,60,128],  # extra label for opening (door&window)
	10: [ 0, 0,  0]  # extra label for wall line
}

def rgb2ind(im, color_map=floorplan_room_map):
	ind = np.zeros((im.shape[0], im.shape[1]))

	for i, rgb in color_map.items():
		ind[(im==rgb).all(2)] = i

	# return ind.astype(int) # int => int64
	return ind.astype(np.uint8) # force to uint8

def ind2rgb(ind_im, color_map=floorplan_room_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

	for i, rgb in color_map.items():
		rgb_im[(ind_im==i)] = rgb

	return rgb_im

def unscale_imsave(path, im, cmin=0, cmax=255):
	toimage(im, cmin=cmin, cmax=cmax).save(path)