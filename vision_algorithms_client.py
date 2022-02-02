import argparse
import logging
import grpc
import vision_algorithms_pb2
import vision_algorithms_pb2_grpc
import numpy as np


# def parse_args():
#     parser = argparse.ArgumentParser(description='Client for testing Vision Algorithms component.')
#     parser.add_argument('--algorithm', metavar='algorithm',
#                         help='Algorithm to test (homog_calc, klt, etc)')
#
#     return parser.parse_args()

def array_to_msg(line):
    """Returns a message of type Float1DArray with elements in list-like argument."""
    array_msg = vision_algorithms_pb2.Float1DArray()
    array_msg.elems.extend(line.tolist())
    return array_msg


def matrix_to_msg(matrix):
    """Returns a message of type Float2DArray with elements from the input two-dimensional array."""
    matrix_msg = vision_algorithms_pb2.Float2DArray()
    if matrix is None:
        return matrix_msg
    for i in range(matrix.shape[0]):
        matrix_msg.lines.append(array_to_msg(matrix[i, :]))
    return matrix_msg


def msg_to_matrix(msg, n_lines=None, n_cols=None):
    """Decodes a 2D array from a protobuf message and returns it as numpy array.

    :param msg: Message of type Float2DArray
    :param n_lines: (Optional) Length of array. If not indicated, it is inferred from the message.
    :param n_cols: (Optional) Width of array. If not indicated, it is inferred from the first line of the array.
    :return: Numpy array
    """
    # Get input data size
    if n_lines is None or n_cols is None:
        n_lines = len(msg.lines)
        n_cols = len(msg.lines[0].elems)

    # Build array from message
    matrix = np.zeros((n_lines, n_cols))
    for i in range(n_lines):
        if len(msg.lines[i].elems) != n_cols:  # Detect line with more than n_cols elements
            logging.error("Detected inconsistent number of elements per line.")
            return None
        matrix[i] = np.array(msg.lines[i].elems)
    return matrix


def get_request_homog_calc(matrix1, matrix2):
    """Returns a message of type HomogCalcRequest.

    Keyword arguments:
    matrix1 -- source points to append to message
    matrix2 -- destination points to append to message
    """
    matrix1_msg = matrix_to_msg(matrix1)
    matrix2_msg = matrix_to_msg(matrix2)
    return vision_algorithms_pb2.HomogCalcRequest(source_pts=matrix1_msg, dest_pts=matrix2_msg, ransac_thresh=5, max_ransac_iters=200)


def homog_calc(stub):
    # Generate ground truth homography
    rotation = np.array([[np.cos(0.7), -np.sin(0.7), 0],
                         [np.sin(0.7), np.cos(0.7), 0],
                         [0, 0, 1]])
    affine = np.array([[1, 0.1, 0],
                       [0.2, 1, 0],
                       [0, 0, 1]])
    projective = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0.05, 0.05, 1]])
    true_homog = rotation @ affine @ projective

    # Generate source and destination points, according to ground truth homography
    src_pts = np.random.rand(2, 5) * 100
    dst_pts = true_homog @ np.r_[src_pts, np.ones((1, src_pts.shape[1]))]
    dst_pts[0, :] = dst_pts[0, :] / dst_pts[2, :]
    dst_pts[1, :] = dst_pts[1, :] / dst_pts[2, :]
    dst_pts = np.delete(dst_pts, 2, 0)

    homog_request = get_request_homog_calc(src_pts.T, dst_pts.T)
    return stub.Process(vision_algorithms_pb2.ExecRequest(homog_calc_args=homog_request))


if __name__ == '__main__':
    # args = parse_args()
    with grpc.insecure_channel('localhost:50051') as channel:
        estimator_stub = vision_algorithms_pb2_grpc.VisionAlgorithmsStub(channel)
        try:
            response = homog_calc(estimator_stub)
            print("Client: Received homography ", msg_to_matrix(response.homog_calc_out.homography))
        except grpc.RpcError as rpc_error:
            print('An error has occurred:')
            print(f'  Error Code: {rpc_error.code()}')
            print(f'  Details: {rpc_error.details()}')
