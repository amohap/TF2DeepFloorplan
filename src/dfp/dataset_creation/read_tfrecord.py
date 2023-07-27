"""
Please prepare the raw image datas save to one folder, 
makesure the path is match to the train_file/test_file.
"""
from tf_record import *


# debug
if __name__ == '__main__':
	
    # read from TFRecord
    loader_list = read_record_act_furn('/media/amohap/Crucial \X8/dataset/Structured3D_TF2Deep/tf2deep_act_furn_train.tfrecords')


    images = loader_list['images']
    walls = loader_list['walls']
    doors = loader_list['doors']
    rooms = loader_list['rooms']
    furns = loader_list['furns']

    with tf.Session() as sess:
	# init all variables in graph
        sess.run(tf.group(tf.global_variables_initializer(),
	 					tf.local_variables_initializer()))
		
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        image, wall, door, room, furn = sess.run([images, walls, doors, rooms, furns])

        print('sess run image shape = ',image.shape)
        print('sess run wall shape = ', wall.shape)
        print('sess run room shape =', room.shape)
        print('sess run door shape =', door.shape)
        print('sess run furn shape =', furn.shape)

        room = np.argmax(np.squeeze(room), axis=-1)
        furn = np.argmax(np.squeeze(furn), axis=-1)
        plt.subplot(231)
        plt.imshow(np.squeeze(image))
        plt.subplot(233)
        plt.imshow(room)
        plt.subplot(234)
        plt.imshow(furn)
        plt.show()

        coord.request_stop()
        coord.join(threads)
        sess.close()