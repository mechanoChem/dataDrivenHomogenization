import tensorflow as tf
import numpy as np

def get_mask_tensor_with_inner_zeros(shape=(5,5), dtype=tf.float32):
    """ create a m*n*1 tensor with the inner region to be zero """
    mask_tensor = np.zeros(shape)
    mask_tensor[0,:] = 1
    mask_tensor[-1,:] = 1
    mask_tensor[:,0] = 1
    mask_tensor[:,-1] = 1
    mask_tensor = np.expand_dims(mask_tensor, axis=2)
    mask_tensor = tf.convert_to_tensor(mask_tensor, dtype=dtype)
    return mask_tensor

if __name__ == '__main__':

    data = np.load('numpy-3-61-61-1.npy')
    data = data[0:3,:,:,:]
    # np.save('numpy.vtk', data)
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    mask_tensor = get_mask_tensor_with_inner_zeros(shape=(5,5))
    print("small mask(5*5): ", mask_tensor[:,:,0])
    mask_tensor = get_mask_tensor_with_inner_zeros(shape=(61,61))
    print("size of mask_tensor:", np.shape(mask_tensor))

    output = tf.math.multiply(data, mask_tensor)

    print("data size: ", np.shape(data))
    print("output size: ", np.shape(output))
    print(data[0,:,:,0])
    print(output[0,:,:,0])



