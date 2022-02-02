import logging
import grpc
import numpy as np

import vision_algorithms_pb2
import vision_algorithms_pb2_grpc
import parsing

# def parse_args():
#     parser = argparse.ArgumentParser(description='Client for testing Vision Algorithms component.')
#     parser.add_argument('--algorithm', metavar='algorithm',
#                         help='Algorithm to test (homog_calc, klt, etc)')
#
#     return parser.parse_args()


def get_request_homog_calc(matrix1, matrix2):
    """Returns a message of type HomogCalcRequest.

    Keyword arguments:
    matrix1 -- source points to append to message
    matrix2 -- destination points to append to message
    """
    matrix1_msg = parsing.matrix_to_msg(matrix1)
    matrix2_msg = parsing.matrix_to_msg(matrix2)
    return vision_algorithms_pb2.HomogCalcRequest(source_pts=matrix1_msg, dest_pts=matrix2_msg, ransac_thresh=5,
                                                  max_ransac_iters=200)


def homog_calc_test(stub):
    # Generate ground truth homography
    angle = 0.1
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    a = 0
    b = 0
    affine = np.array([[1, a, 0],
                       [b, 1, 0],
                       [0, 0, 1]])
    c = 0
    projective = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [c, c, 1]])
    true_homog = rotation @ affine @ projective

    # Generate source and destination points, according to ground truth homography
    src_pts = np.random.rand(2, 5) * 100
    dst_pts = true_homog @ np.r_[src_pts, np.ones((1, src_pts.shape[1]))]
    dst_pts[0, :] = dst_pts[0, :] / dst_pts[2, :]
    dst_pts[1, :] = dst_pts[1, :] / dst_pts[2, :]
    dst_pts = np.delete(dst_pts, 2, 0)

    homog_request = get_request_homog_calc(src_pts.T, dst_pts.T)
    return stub.Process(vision_algorithms_pb2.ExecRequest(homog_calc_args=homog_request))


def homog_warp_test(stub, homog_msg):
    img = np.random.random_sample((3, 300, 600))*255
    img_msg = parsing.image_to_msg(img)
    warp_request = vision_algorithms_pb2.HomogWarpRequest(image=img_msg, homography=homog_msg, out_width=10,
                                                          out_height=10)
    return stub.Process(vision_algorithms_pb2.ExecRequest(homog_warp_args=warp_request))


if __name__ == '__main__':
    # args = parse_args()
    with grpc.insecure_channel('localhost:50051') as channel:
        estimator_stub = vision_algorithms_pb2_grpc.VisionAlgorithmsStub(channel)
        try:
            response = homog_calc_test(estimator_stub)
            print("Client: Received homography ", parsing.msg_to_matrix(response.homog_calc_out.homography))
            response = homog_warp_test(estimator_stub, response.homog_calc_out.homography)
            print("Client: Received warped image.")
        except grpc.RpcError as rpc_error:
            print('An error has occurred:')
            print(f'  Error Code: {rpc_error.code()}')
            print(f'  Details: {rpc_error.details()}')
